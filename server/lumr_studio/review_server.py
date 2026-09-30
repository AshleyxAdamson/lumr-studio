"""A small local web server for the treatment page, standard library only.

The creator tunes Claude's edit in their browser: ``url_for(project)`` gives
them a link, the page plays the edit straight from the source video, and
their changes come back here to be applied to the edit (see ``treatment.py``).
Tuning renders nothing. Export video starts the same full render Claude's
render tool starts, in the same job registry (see ``render_jobs.py``).

Security, least privilege:

* Binds 127.0.0.1 only, on a port the OS picks.
* Every route sits under a random per-project token made at registration
  (``secrets.token_urlsafe``); a wrong token gets 404.
* A ``Host`` header other than ``127.0.0.1:<port>`` or ``localhost:<port>`` is
  refused, which stops DNS rebinding. A POST from another origin is refused.
* POST bodies must be ``application/json`` and under ``MAX_BODY_BYTES``.
* The video route serves only the project's own video; no path ever comes
  from the request. The page is sent with a Content-Security-Policy that
  allows no external requests.

Routes, all under ``/<token>/``. Every treatment POST answers the whole new
state; the shapes are in the round's API notes and ``treatment.page_state``.

    GET  /                the page: treatment_page/index.html
    GET  api/treatment    the treatment state (treatment.page_state)
    POST api/treatment    {pace?, take_out?}: change the pace and switches
    POST api/cut          {id, state}: put one of Claude's cuts back or keep it out
    POST api/rate         {id, rating}: rate one of Claude's cuts good or bad (null clears); bad also puts it back
    POST api/cut/add      {start, end}: the creator cuts exactly these words
    POST api/cut/remove   {id, start?, end?}: take back a cut the creator made, or part of it
    POST api/keep         {start, end, exact?, note?}: keep a stretch no matter what
    POST api/keep/remove  {id}: remove a kept stretch
    POST api/undo         {}: take back the last thing the creator did to the words
    POST api/samples      {}: pick three new samples
    POST api/usual        {}: save the pace and switches as the creator's usual
    POST api/export       {}: start the full render, unless one is running
    GET  api/export       the latest export: {export: {state, progress, file, ...}}
    GET  api/peaks        the audio envelope (the page no longer asks for it)
    GET  video            the source video, with HTTP Range support
    GET  frame?t=<s>      one JPEG still, cached under <project.root>/review/frames/

The older review page is gone; its routes still answer, and nothing in the
plugin calls them now:

    GET  api/queue        rows, counts and durations (review.queue_for_project)
    POST api/decisions    save decisions to <project.root>/review.json
    POST api/apply        apply the saved decisions to the edit
"""

from __future__ import annotations

import base64
import json
import logging
import math
import mimetypes
import os
import secrets
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from lumr_studio import review
from lumr_studio import treatment
from lumr_studio.edit import clock, load_edit
from lumr_studio.errors import StudioError
from lumr_studio.jobs import JOBS, JobRegistry
from lumr_studio.project import Project
from lumr_studio.render_jobs import RenderWork, export_status, start_full_render
from lumr_studio.silences import SilenceProvider, measured_silences

log = logging.getLogger(__name__)

BIND_HOST = "127.0.0.1"
PAGE_PATH = Path(__file__).parent / "treatment_page" / "index.html"
# The page's decisions for a few hundred cuts are a few kilobytes; a megabyte
# is far more than any honest request needs.
MAX_BODY_BYTES = 1_000_000
# Video is streamed in chunks this size so a seek never loads the whole file.
CHUNK_BYTES = 256 * 1024
# Frame stills: small enough that dozens load fast, big enough to see a head move.
FRAME_WIDTH = 320
FRAME_QUALITY = 4  # ffmpeg -q:v, 2 (best) to 31 (worst)
# Stills are keyed to the hundredth of a second, the same precision as row ids.
FRAME_KEY_SCALE = 100
# At most this many ffmpeg processes grab stills at once, so a page full of
# flagged rows can't start hundreds of decoders.
FRAME_WORKERS = 3
FFMPEG_TIMEOUT = 30
# How often the serve loop checks for stop(); also how long stop() can wait.
POLL_SECONDS = 0.1
# The wave drawings use one peak per 20 ms: a row's 5-second snippet is 108px
# wide, so this is finer than a pixel, and a 22-minute video is 66k bytes.
PEAK_SECONDS = 0.02
PEAK_SAMPLE_RATE = 8000
# Peaks are scaled so this quantile of the loud parts reads as full height;
# a single shout would otherwise flatten everything else.
PEAK_REFERENCE_QUANTILE = 0.995

PAGE_CSP = (
    "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
    "img-src 'self'; media-src 'self'; connect-src 'self'; "
    "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)

JoinRows = Callable[..., list[dict[str, Any]]]
LabelsFor = Callable[[Project, list[dict[str, Any]]], list[dict[str, Any]]]
FrameGrabber = Callable[[Path, float, Path], None]
PeaksMeasure = Callable[[Path], bytes]


# ── Media helpers (production collaborators; tests may pass fakes) ───────────


def grab_frame(video: Path, t: float, out_path: Path) -> None:
    """Write one JPEG still of ``video`` at source time ``t`` to ``out_path``.

    Uses input seeking, so it costs the same at minute 20 as at second 1.
    Raises StudioError when ffmpeg fails or writes nothing.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".frame.", suffix=".jpg", dir=out_path.parent)
    os.close(fd)
    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(video),
                "-frames:v", "1", "-vf", f"scale={FRAME_WIDTH}:-2", "-q:v", str(FRAME_QUALITY),
                "-f", "image2", tmp,
            ],
            capture_output=True, text=True, timeout=FFMPEG_TIMEOUT,
        )
        if proc.returncode != 0 or os.path.getsize(tmp) == 0:
            raise StudioError(
                f"ffmpeg could not read a frame at {t:.2f}s ({proc.stderr.strip()[:200] or 'no output'}). "
                "Check that ffmpeg is installed and the video plays."
            )
        os.replace(tmp, out_path)
    except subprocess.TimeoutExpired:
        raise StudioError(f"ffmpeg took over {FFMPEG_TIMEOUT}s to read a frame at {t:.2f}s. Try again.") from None
    finally:
        Path(tmp).unlink(missing_ok=True)


def measure_peaks(video: Path) -> bytes:
    """The audio envelope of ``video``: one byte (0-255) per ``PEAK_SECONDS``.

    Decodes the audio once with ffmpeg at a low sample rate and takes the peak
    of each window with numpy. Raises StudioError when ffmpeg fails.
    """
    import numpy as np

    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-v", "error", "-i", str(video), "-vn", "-ac", "1",
                "-ar", str(PEAK_SAMPLE_RATE), "-f", "s16le", "-",
            ],
            capture_output=True, timeout=600,
        )
    except subprocess.TimeoutExpired:
        raise StudioError("ffmpeg took too long to read the audio. Try again.") from None
    if proc.returncode != 0:
        raise StudioError(
            f"ffmpeg could not read the audio ({proc.stderr.decode(errors='replace').strip()[:200]}). "
            "Check that the video has a sound track and ffmpeg is installed."
        )
    samples = np.abs(np.frombuffer(proc.stdout, dtype=np.int16).astype(np.int32))
    per = int(PEAK_SAMPLE_RATE * PEAK_SECONDS)
    n = len(samples) // per
    if n == 0:
        return b""
    peaks = samples[: n * per].reshape(n, per).max(axis=1).astype(np.float64)
    loud = peaks[peaks > 0]
    ref = float(np.quantile(loud, PEAK_REFERENCE_QUANTILE)) if loud.size else 1.0
    scaled = np.clip(peaks / max(ref, 1.0), 0.0, 1.0)
    return (np.round(scaled * 255)).astype(np.uint8).tobytes()


def _default_join_rows(cuts: list[dict[str, Any]], words: list[dict[str, Any]], duration: float, *,
                       labels: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """The join self-check from ``lumr_studio.joins``, imported when first needed."""
    try:
        from lumr_studio.joins import join_rows
    except ImportError:
        raise StudioError(
            "The join self-check (lumr_studio/joins.py) is missing, so the review page can't show "
            "flags. Reinstall the plugin."
        ) from None
    return join_rows(cuts, words, duration, labels=labels)


# ── Range requests ───────────────────────────────────────────────────────────

UNSATISFIABLE = "unsatisfiable"


def parse_range(header: str | None, size: int) -> tuple[int, int] | str | None:
    """The byte span a ``Range`` header asks for, inclusive, within a file of ``size`` bytes.

    Returns ``(first, last)``, ``UNSATISFIABLE`` when the range starts past the
    end, or None when the header is absent, malformed or asks for several
    ranges (the whole file is sent then, as HTTP allows).
    """
    if not header:
        return None
    unit, _, spec = header.strip().partition("=")
    if unit.strip().lower() != "bytes" or "," in spec:
        return None
    first_s, dash, last_s = spec.strip().partition("-")
    if not dash:
        return None
    try:
        if first_s == "":
            suffix = int(last_s)
            if suffix < 0:
                return None
            if suffix == 0 or size == 0:
                return UNSATISFIABLE
            return max(0, size - suffix), size - 1
        first = int(first_s)
        last = int(last_s) if last_s else size - 1
    except ValueError:
        return None
    if first < 0 or (last_s and last < first):
        return None
    if first >= size:
        return UNSATISFIABLE
    return first, min(last, size - 1)


# ── Per-project state ────────────────────────────────────────────────────────


@dataclass
class _Session:
    project: Project
    token: str
    duration: float
    # Serializes the reads and writes of the edit for one project.
    lock: threading.Lock = field(default_factory=threading.Lock)
    # Held while the audio envelope is measured, which takes seconds; kept
    # apart from ``lock`` so saving decisions never waits on it.
    peaks_lock: threading.Lock = field(default_factory=threading.Lock)
    peaks: bytes | None = None

    @property
    def frames_dir(self) -> Path:
        return self.project.root / "review" / "frames"


class _HTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    review: "ReviewServer"

    def handle_error(self, request: Any, client_address: Any) -> None:
        """Log a failed request, but stay quiet when the browser simply dropped the connection.

        A video element closes kept-alive connections on every seek. The
        standard library prints a full traceback for each one, which buried
        real errors under dozens of resets in one page session.
        """
        if isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError)):
            return
        log.exception("review server: a request from %s failed", client_address)


class ReviewServer:
    """The review page's local server. One per plugin process; many projects.

    Collaborators that touch the outside world are keyword parameters so tests
    can pass fakes: ``join_rows`` (the join self-check), ``labels_for`` (sound
    labels), ``silences`` (measured silences for word-safe edge placement),
    ``grab_frame`` and ``measure_peaks`` (ffmpeg), ``jobs`` (the registry an
    export runs in, the tools' own by default so job_status sees it) and
    ``render`` (the work of an export; None is the real render).
    """

    def __init__(
        self,
        *,
        join_rows: JoinRows = _default_join_rows,
        labels_for: LabelsFor | None = None,
        silences: SilenceProvider = measured_silences,
        grab_frame: FrameGrabber = grab_frame,
        measure_peaks: PeaksMeasure = measure_peaks,
        jobs: JobRegistry = JOBS,
        render: RenderWork | None = None,
    ) -> None:
        self.jobs = jobs
        self.render = render
        self.join_rows = join_rows
        self.labels_for = labels_for
        self.silences = silences
        self.grab_frame = grab_frame
        self.measure_peaks = measure_peaks
        self._httpd: _HTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._sessions: dict[str, _Session] = {}
        self._by_root: dict[Path, str] = {}
        self._register = threading.Lock()
        self._frame_slots = threading.BoundedSemaphore(FRAME_WORKERS)

    # lifecycle

    def start(self) -> None:
        """Bind 127.0.0.1 on a free port and serve on a daemon thread. A second call does nothing."""
        if self._httpd is not None:
            return
        httpd = _HTTPServer((BIND_HOST, 0), _Handler)
        httpd.review = self
        self._httpd = httpd
        self._thread = threading.Thread(
            target=httpd.serve_forever, kwargs={"poll_interval": POLL_SECONDS}, name="lumr-review", daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop serving and release the port. Safe to call twice."""
        httpd, self._httpd = self._httpd, None
        if httpd is None:
            return
        httpd.shutdown()
        httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._thread = None

    @property
    def port(self) -> int:
        if self._httpd is None:
            raise RuntimeError("ReviewServer.start() must be called first.")
        return self._httpd.server_address[1]

    def url_for(self, project: Project) -> str:
        """The page's address for ``project``: ``http://127.0.0.1:<port>/<token>/``.

        The token is made on the first call for a project and reused after, for
        as long as this server runs. Raises StudioError when the video's length
        can't be read.
        """
        port = self.port
        with self._register:
            token = self._by_root.get(project.root)
            if token is None:
                token = secrets.token_urlsafe(24)
                self._sessions[token] = _Session(project=project, token=token, duration=project.duration())
                self._by_root[project.root] = token
        return f"http://{BIND_HOST}:{port}/{token}/"

    # used by the handler

    def session_for(self, token: str) -> _Session | None:
        for known, session in list(self._sessions.items()):
            if secrets.compare_digest(known, token):
                return session
        return None

    def allowed_hosts(self) -> set[str]:
        port = self.port
        return {f"127.0.0.1:{port}", f"localhost:{port}"}

    def frame_path(self, session: _Session, t: float) -> Path:
        """The cached still for time ``t``, grabbed on first request."""
        key = int(round(t * FRAME_KEY_SCALE))
        path = session.frames_dir / f"{key:08d}.jpg"
        if not path.exists():
            with self._frame_slots:
                if not path.exists():
                    self.grab_frame(session.project.video, key / FRAME_KEY_SCALE, path)
        return path

    def treatment_context(self, session: _Session) -> treatment.Context:
        """The project's edit, transcript, silences and labels, loaded fresh for one request."""
        return treatment.load_context(
            session.project, duration=session.duration, silences=self.silences,
            labels_for=self.labels_for, join_rows=self.join_rows,
            export_for=lambda edit: self.export_of(session, edit),
        )

    def export_of(self, session: _Session, edit: dict[str, Any] | None = None) -> dict[str, Any]:
        """The latest export of the session's video, read against ``edit`` (the saved edit when left out).

        Takes no lock: the edit file is replaced whole on every save, and the
        page asks once a second while an export runs.
        """
        if edit is None:
            edit = load_edit(session.project, session.duration)
        return export_status(session.project, self.jobs, edit=edit)

    def start_export(self, session: _Session) -> dict[str, Any]:
        """Start the full render of the saved edit and answer the export. One runs per video.

        While one is running, started here or by Claude's render tool, this
        starts nothing and answers the running one.
        """
        start_full_render(session.project, jobs=self.jobs, duration=session.duration, render=self.render)
        return self.export_of(session)

    def peaks(self, session: _Session) -> bytes:
        """The audio envelope, measured once per session and cached on disk by video size and time."""
        if session.peaks is None:
            with session.peaks_lock:
                if session.peaks is None:
                    session.peaks = self._cached_peaks(session)
        return session.peaks

    def _cached_peaks(self, session: _Session) -> bytes:
        video = session.project.video
        stat = video.stat()
        key = f"{stat.st_size}-{int(stat.st_mtime)}-{PEAK_SECONDS}"
        cache = session.project.root / "review" / f"peaks-{key}.bin"
        if cache.exists():
            return cache.read_bytes()
        data = self.measure_peaks(video)
        cache.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".peaks.", dir=cache.parent)
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, cache)
        return data


# ── The handler ──────────────────────────────────────────────────────────────

# Each treatment POST route and the action that answers it. Every action
# validates its body and raises StudioError, a 400, with what to fix.
TREATMENT_ACTIONS: dict[str, Callable[[treatment.Context, dict[str, Any]], dict[str, Any]]] = {
    "api/treatment": treatment.change_treatment,
    "api/cut": treatment.set_cut_state,
    "api/rate": treatment.rate_cut,
    "api/cut/add": treatment.add_cut,
    "api/cut/remove": treatment.remove_cut,
    "api/keep": treatment.add_keep,
    "api/keep/remove": treatment.remove_keep,
    "api/undo": treatment.undo,
    "api/samples": treatment.new_samples,
    "api/usual": treatment.save_as_usual,
}


# Export is apart from the treatment actions: it changes nothing in the edit
# and answers the export alone, since the page asks for it once a second.
EXPORT_ROUTE = "api/export"


class _Refused(Exception):
    """A request the server turns away. Carries the status and a plain message."""

    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class _Redirect(Exception):
    """Send the browser to ``location`` on this server (the page needs its trailing slash)."""

    def __init__(self, location: str) -> None:
        super().__init__(location)
        self.location = location


class _Handler(BaseHTTPRequestHandler):
    server: _HTTPServer
    protocol_version = "HTTP/1.1"
    server_version = "LumrReview"
    sys_version = ""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        # stdout belongs to the MCP transport; keep request logs at debug level.
        log.debug("review %s - %s", self.address_string(), format % args)

    # entry points

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_HEAD(self) -> None:
        self._dispatch("HEAD")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def _dispatch(self, method: str) -> None:
        try:
            self._check_host()
            session, route, query = self._route()
            if method in ("GET", "HEAD"):
                self._get(session, route, query, head=method == "HEAD")
            else:
                self._post(session, route)
        except _Redirect as to:
            self.send_response(HTTPStatus.PERMANENT_REDIRECT)
            self.send_header("Location", to.location)
            self.send_header("Content-Length", "0")
            self._common_headers("no-store")
            self.end_headers()
        except _Refused as why:
            self._send_json({"error": why.message}, why.status)
        except StudioError as why:
            self._send_json({"error": str(why)}, HTTPStatus.BAD_REQUEST)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the browser gave up on this request, which it does on every seek
        except Exception:
            log.exception("review server failed on %s %s", method, self.path)
            self._send_json(
                {"error": "The review server hit an unexpected error. Reload the page; if it keeps happening, restart it."},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    # checks

    def _check_host(self) -> None:
        if self.headers.get("Host", "") not in self.server.review.allowed_hosts():
            raise _Refused(HTTPStatus.FORBIDDEN, "This address is not allowed. Open the link Claude gave you.")

    def _route(self) -> tuple[_Session, str, dict[str, list[str]]]:
        parts = urlsplit(self.path)
        segments = parts.path.split("/", 2)  # ['', token, rest]
        token = segments[1] if len(segments) > 1 else ""
        session = self.server.review.session_for(token) if token else None
        if session is None:
            raise _Refused(HTTPStatus.NOT_FOUND, "Not found. This review link has expired; ask Claude for a new one.")
        if len(segments) < 3:
            raise _Redirect(f"/{session.token}/")
        return session, segments[2], parse_qs(parts.query)

    # GET

    def _get(self, session: _Session, route: str, query: dict[str, list[str]], *, head: bool) -> None:
        review_server = self.server.review
        if route == "":
            if not PAGE_PATH.exists():
                raise _Refused(HTTPStatus.INTERNAL_SERVER_ERROR, "The page file is missing. Reinstall the plugin.")
            body = PAGE_PATH.read_bytes()
            self._send_bytes(body, "text/html; charset=utf-8", head=head, extra={"Content-Security-Policy": PAGE_CSP})
        elif route == "api/treatment":
            with session.lock:
                state = treatment.page_state(review_server.treatment_context(session))
            self._send_json(state, head=head)
        elif route == EXPORT_ROUTE:
            self._send_json({"export": review_server.export_of(session)}, head=head)
        elif route == "api/queue":
            queue = review.queue_for_project(
                session.project, join_rows=review_server.join_rows,
                labels_for=review_server.labels_for, duration=session.duration,
            )
            self._send_json(queue, head=head)
        elif route == "api/peaks":
            data = review_server.peaks(session)
            self._send_json({"seconds_per_peak": PEAK_SECONDS, "peaks": base64.b64encode(data).decode("ascii")}, head=head)
        elif route == "video":
            self._send_video(session.project.video, head=head)
        elif route == "frame":
            t = self._frame_time(query, session.duration)
            path = review_server.frame_path(session, t)
            self._send_bytes(path.read_bytes(), "image/jpeg", head=head, cache="private, max-age=86400")
        else:
            raise _Refused(HTTPStatus.NOT_FOUND, "Not found.")

    @staticmethod
    def _frame_time(query: dict[str, list[str]], duration: float) -> float:
        raw = (query.get("t") or [""])[0]
        try:
            t = float(raw)
        except ValueError:
            raise _Refused(HTTPStatus.BAD_REQUEST, "frame needs t, a number of seconds, like frame?t=12.5.") from None
        if not math.isfinite(t) or t < 0 or t > duration:
            raise _Refused(HTTPStatus.BAD_REQUEST, f"t={raw} is outside the video, which runs from 0:00 to {clock(duration)}.")
        # The last frame starts a little before the end; ffmpeg finds nothing past it.
        return min(t, max(0.0, duration - 0.1))

    def _send_video(self, video: Path, *, head: bool) -> None:
        size = video.stat().st_size
        ctype = mimetypes.guess_type(video.name)[0] or "application/octet-stream"
        wanted = parse_range(self.headers.get("Range"), size)
        if wanted == UNSATISFIABLE:
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self._common_headers("no-store")
            self.end_headers()
            return
        if wanted is None:
            first, last, status = 0, size - 1, HTTPStatus.OK
        else:
            first, last = wanted
            status = HTTPStatus.PARTIAL_CONTENT
        length = max(0, last - first + 1)
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {first}-{last}/{size}")
        self._common_headers("private, max-age=3600")
        self.end_headers()
        if head:
            return
        with open(video, "rb") as fh:
            fh.seek(first)
            remaining = length
            while remaining > 0:
                chunk = fh.read(min(CHUNK_BYTES, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    # POST

    def _post(self, session: _Session, route: str) -> None:
        if route not in ("api/decisions", "api/apply", EXPORT_ROUTE, *TREATMENT_ACTIONS):
            raise _Refused(HTTPStatus.NOT_FOUND, "Not found.")
        body = self._json_body()
        review_server = self.server.review
        with session.lock:
            if route in TREATMENT_ACTIONS:
                result = TREATMENT_ACTIONS[route](review_server.treatment_context(session), body)
            elif route == EXPORT_ROUTE:
                if body:
                    raise _Refused(HTTPStatus.BAD_REQUEST, "Export takes no settings. Send an empty JSON object, {}.")
                result = {"export": review_server.start_export(session)}
            elif route == "api/decisions":
                if "decisions" not in body:
                    raise _Refused(HTTPStatus.BAD_REQUEST, 'Send {"decisions": {row id: {state, start, end}}}.')
                result = review.save_decisions(session.project, body["decisions"], duration=session.duration)
            else:
                silences = review_server.silences(session.project.video)
                result = review.apply_to_project(session.project, silences=silences, duration=session.duration)
        self._send_json(result)

    def _json_body(self) -> dict[str, Any]:
        origin = self.headers.get("Origin")
        if origin is not None and urlsplit(origin).netloc not in self.server.review.allowed_hosts():
            raise _Refused(HTTPStatus.FORBIDDEN, "Requests from other sites are not allowed.")
        ctype = self.headers.get("Content-Type", "").split(";")[0].strip().lower()
        if ctype != "application/json":
            raise _Refused(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Send the body as JSON with Content-Type: application/json.")
        length_raw = self.headers.get("Content-Length")
        if length_raw is None:
            raise _Refused(HTTPStatus.LENGTH_REQUIRED, "Send a Content-Length header.")
        try:
            length = int(length_raw)
        except ValueError:
            raise _Refused(HTTPStatus.BAD_REQUEST, "Content-Length must be a whole number.") from None
        if length < 0 or length > MAX_BODY_BYTES:
            self.close_connection = True
            raise _Refused(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"The body must be under {MAX_BODY_BYTES} bytes.")
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise _Refused(HTTPStatus.BAD_REQUEST, "The body is not valid JSON.") from None
        if not isinstance(body, dict):
            raise _Refused(HTTPStatus.BAD_REQUEST, "The body must be a JSON object.")
        return body

    # responses

    def _common_headers(self, cache: str) -> None:
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")

    def _send_bytes(self, body: bytes, ctype: str, *, head: bool = False, status: HTTPStatus = HTTPStatus.OK,
                    cache: str = "no-store", extra: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self._common_headers(cache)
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if not head:
            self.wfile.write(body)

    def _send_json(self, data: Any, status: HTTPStatus = HTTPStatus.OK, *, head: bool = False) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self._send_bytes(body, "application/json; charset=utf-8", head=head, status=status)


# ── One shared server per plugin process ─────────────────────────────────────

_shared: ReviewServer | None = None
_shared_lock = threading.Lock()


def review_url(project: Project, **collaborators: Any) -> str:
    """Start the process-wide review server on first use and return ``project``'s page address.

    ``collaborators`` go to ``ReviewServer`` on the first call only.
    """
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = ReviewServer(**collaborators)
            _shared.start()
        server = _shared
    return server.url_for(project)
