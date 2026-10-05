"""Build a fake UI feedback screen recording with known answers.

Draws four "screens" of a made-up web app, holds each for a set time the way a
macOS screen recording does (frames only when the screen changes, so the video
is variable frame rate), and lays spoken feedback over it with macOS `say`, each
line starting at a known second. Writes:

    <out>/feedback-demo.mp4   the recording
    <out>/answers.json        each line: start time, text, which screen is up

Usage (from any folder; needs ffmpeg, Pillow, and macOS `say`):

    python make_feedback_recording.py <out folder>
    python make_feedback_recording.py <out folder> --tones   # no `say`: beeps instead of words
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 2880, 1800  # a retina MacBook screen, so the tool has to scale down

# (screen name, seconds on screen). Total 40 s.
SCREENS = [("home", 10.0), ("pricing", 12.0), ("signup", 10.0), ("settings", 8.0)]

# (start second, words). Each line starts while the named screen is up.
LINES = [
    (1.0, "Okay, this is the home page. The headline is great, I really like it."),
    (5.5, "But this grey text under it is way too light, I can barely read it."),
    (11.0, "Now pricing. These three cards are fine."),
    (15.0, "The middle card should be highlighted, it's the one we want people to pick."),
    (19.5, "And this button here says Buy. Change it to Start free trial."),
    (23.0, "Sign up. The email field is tiny, make it full width."),
    (28.0, "Also there's no way to see my password. Add a show password toggle."),
    (33.0, "Settings looks good. No notes here."),
]


def font(size: int) -> ImageFont.ImageFont:
    for path in ("/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


def nav(d: ImageDraw.ImageDraw, active: str) -> None:
    d.rectangle([0, 0, W, 140], fill=(250, 250, 250))
    d.text((80, 40), "Acme", fill=(20, 20, 20), font=font(64))
    for i, name in enumerate(["Home", "Pricing", "Sign up", "Settings"]):
        x = 1400 + i * 340
        d.text((x, 50), name, fill=(0, 90, 220) if name.lower().replace(" ", "") == active else (60, 60, 60), font=font(44))


def screen(name: str) -> Image.Image:
    im = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(im)
    nav(d, name)
    if name == "home":
        d.text((200, 500), "Ship videos faster", fill=(10, 10, 10), font=font(150))
        d.text((200, 720), "Cut the gaps, keep the jokes, export in minutes.", fill=(205, 205, 205), font=font(56))
    elif name == "pricing":
        for i, (plan, price) in enumerate([("Starter", "$0"), ("Pro", "$12"), ("Team", "$40")]):
            x = 240 + i * 820
            d.rounded_rectangle([x, 400, x + 700, 1500], radius=40, outline=(200, 200, 200), width=6)
            d.text((x + 60, 460), plan, fill=(20, 20, 20), font=font(80))
            d.text((x + 60, 600), price, fill=(20, 20, 20), font=font(120))
            d.rounded_rectangle([x + 60, 1300, x + 640, 1420], radius=30, fill=(0, 90, 220))
            d.text((x + 290, 1325), "Buy", fill="white", font=font(60))
    elif name == "signup":
        d.text((200, 400), "Create your account", fill=(10, 10, 10), font=font(100))
        d.text((200, 640), "Email", fill=(40, 40, 40), font=font(48))
        d.rectangle([200, 710, 760, 800], outline=(150, 150, 150), width=4)
        d.text((200, 880), "Password", fill=(40, 40, 40), font=font(48))
        d.rectangle([200, 950, 1600, 1040], outline=(150, 150, 150), width=4)
        d.text((230, 965), "••••••••", fill=(40, 40, 40), font=font(48))
    elif name == "settings":
        d.text((200, 400), "Settings", fill=(10, 10, 10), font=font(100))
        for i, row in enumerate(["Notifications", "Dark mode", "Language"]):
            d.text((200, 620 + i * 160), row, fill=(40, 40, 40), font=font(60))
    return im


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL)


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    tones = "--tones" in sys.argv
    if len(args) != 1:
        sys.exit(__doc__)
    out = Path(args[0]).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not tones and not shutil.which("say"):
        sys.exit("macOS `say` not found. Run on a Mac, or pass --tones to check the build without words.")
    total = sum(s for _, s in SCREENS)
    with tempfile.TemporaryDirectory(prefix="feedback-demo-") as tmp:
        work = Path(tmp)
        # Video: one still per screen, held with the concat demuxer: variable frame rate, like a screen recording.
        listing = []
        for name, seconds in SCREENS:
            screen(name).save(work / f"{name}.png")
            listing += [f"file '{work / name}.png'", f"duration {seconds}"]
        listing.append(f"file '{work / SCREENS[-1][0]}.png'")
        (work / "list.txt").write_text("\n".join(listing) + "\n")
        run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(work / "list.txt"),
             "-fps_mode", "vfr", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(work / "video.mp4")])
        # Audio: each line rendered alone, then delayed to its start and mixed.
        inputs, filters = [], []
        for i, (start, text) in enumerate(LINES):
            clip = work / f"line{i}.wav"
            if tones:
                run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i",
                     f"sine=frequency={300 + 60 * i}:duration=2", str(clip)])
            else:
                run(["say", "-o", str(clip), "--data-format=LEI16@22050", text])
            inputs += ["-i", str(clip)]
            filters.append(f"[{i + 1}:a]adelay={int(start * 1000)}:all=1[a{i}]")
        mix = "".join(f"[a{i}]" for i in range(len(LINES)))
        filters.append(f"{mix}amix=inputs={len(LINES)}:normalize=0,apad,atrim=0:{total}[aout]")
        run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(work / "video.mp4"), *inputs,
             "-filter_complex", ";".join(filters), "-map", "0:v", "-map", "[aout]",
             "-c:v", "copy", "-c:a", "aac", "-ar", "48000", str(out / "feedback-demo.mp4")])
    # The answers: which screen is up when each line starts.
    bounds, t = [], 0.0
    for name, seconds in SCREENS:
        bounds.append((t, t + seconds, name))
        t += seconds
    answers = [
        {"start": start, "text": text, "screen": next(n for a, b, n in bounds if a <= start < b)}
        for start, text in LINES
    ]
    (out / "answers.json").write_text(json.dumps({"screens": bounds, "lines": answers}, indent=2) + "\n")
    print(out / "feedback-demo.mp4")


if __name__ == "__main__":
    main()
