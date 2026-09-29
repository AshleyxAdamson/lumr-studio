"""Regression test for the stderr-pipe deadlock in ``clipforge.render.run_render``.

``run_render`` reads ffmpeg's stdout ``-progress`` feed line by line, then
``proc.wait()``s, then reads stderr, and only afterwards. If ffmpeg writes more to
stderr than the OS pipe buffer holds (~64KB on macOS) before it exits, that
write blocks forever with nothing draining it, and the stdout loop (fed by the
same ffmpeg process) goes quiet right along with it: a full deadlock.

No real ffmpeg here: a tiny Python script stands in for it (via ``runner``,
the process-factory seam ``run_render`` already exposes for tests) and floods
stderr with far more than 64KB before finishing, exactly like a real h264
source that trips ffmpeg's "Late SEI is not implemented" warning thousands of
times over a long encode.
"""

import subprocess
import sys
import threading
from pathlib import Path

from lumr_studio.engine.render import RenderPlan, RenderProgress, run_render

# Comfortably past a 64KB OS pipe buffer, so the old (undrained) code reliably
# deadlocks rather than getting lucky with buffering.
STDERR_FLOOD_BYTES = 300_000

FAKE_FFMPEG_SRC = '''
import sys

exit_code = int(sys.argv[1])
out_path = sys.argv[-1]

# A couple of progress lines, same key run_render's _parse_progress_line reads.
print("out_time_ms=1000000", flush=True)
print("out_time_ms=2000000", flush=True)

# Flood stderr well past the OS pipe buffer before the process is done.
chunk = "Late SEI is not implemented near frame N\\n"
written = 0
while written < {flood_bytes} - len(chunk):
    sys.stderr.write(chunk)
    written += len(chunk)
# A recognisable line at the very END of the stream, so a test can tell the
# reported error came from the tail, not a lucky early fragment.
sys.stderr.write("FINAL-ERROR-LINE\\n")
sys.stderr.flush()

if exit_code == 0:
    with open(out_path, "wb") as f:
        f.write(b"\\x00" * 16)

print("progress=end", flush=True)
sys.exit(exit_code)
'''.format(flood_bytes=STDERR_FLOOD_BYTES)

RUN_RENDER_TIMEOUT = 20.0


def _write_fake_ffmpeg(tmp_path: Path) -> Path:
    script = tmp_path / "fake_ffmpeg.py"
    script.write_text(FAKE_FFMPEG_SRC)
    return script


def _plan(tmp_path: Path) -> RenderPlan:
    return RenderPlan(
        source=tmp_path / "source.mp4",  # never touched: the fake ignores it
        out_path=tmp_path / "exports" / "out.mp4",
        kept=[(0.0, 4.0)],
        removed=[],
        duration=4.0,
        out_height=0,
        source_width=0,
        source_height=0,
        source_note="test",
    )


def _make_runner(fake_script: Path, exit_code: int):
    """A ``run_render`` ``runner`` that ignores ffmpeg and launches the fake instead.

    Keeps the same keyword arguments (``stdout``/``stderr``/``text``/``bufsize``)
    ``run_render`` passes, so this exercises the exact same pipe plumbing
    production code uses. ``cmd``'s last element is always the output path (see
    ``build_ffmpeg_cmd``).
    """

    def runner(cmd, **kwargs):
        out_path = cmd[-1]
        return subprocess.Popen(
            [sys.executable, str(fake_script), str(exit_code), out_path],
            **kwargs,
        )

    return runner


def _run_render_with_timeout(plan: RenderPlan, progress: RenderProgress, runner) -> bool:
    """Run ``run_render`` on a thread; True if it returned within the timeout."""
    finished = threading.Event()

    def go():
        run_render(plan, progress, runner=runner)
        finished.set()

    t = threading.Thread(target=go, daemon=True)
    t.start()
    return finished.wait(timeout=RUN_RENDER_TIMEOUT)


def test_stderr_flood_does_not_deadlock(tmp_path):
    fake_script = _write_fake_ffmpeg(tmp_path)
    plan = _plan(tmp_path)
    progress = RenderProgress(job_id="stderr-flood-success")

    returned = _run_render_with_timeout(
        plan, progress, _make_runner(fake_script, exit_code=0)
    )

    assert returned, "run_render did not return within 20s: stderr pipe deadlock"
    assert progress.status == "done"
    assert progress.progress == 1.0
    assert plan.out_path.exists()


def test_stderr_flood_failure_reports_tail_of_stream(tmp_path):
    fake_script = _write_fake_ffmpeg(tmp_path)
    plan = _plan(tmp_path)
    progress = RenderProgress(job_id="stderr-flood-failure")

    returned = _run_render_with_timeout(
        plan, progress, _make_runner(fake_script, exit_code=1)
    )

    assert returned, "run_render did not return within 20s: stderr pipe deadlock"
    assert progress.status == "failed"
    assert not plan.out_path.exists()
    assert "FINAL-ERROR-LINE" in (progress.error or "")
