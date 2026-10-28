"""Render festival/intro_html/intro.html to an MP4 for the 10/28 talk (2026-10-22, user).

Frame-exact: the page draws any time t with window.seek(t), so every frame is a screenshot of
seek(i / FPS) in a headless 1920x1080 Chromium -- no screen recording, no dropped frames.
The page ends on black at 107.5 s; TAIL seconds of black are added so PowerPoint can cut to the
cover slide (dark background) from black. Sound: sfx/sfx_track_v2.wav (78th-move hit only).
Output: festival/intro_video_v2.mp4 (H.264 + AAC, plays in PowerPoint).
Usage: python render_intro_video.py [--short]   (needs internet for Google Fonts; port 8791 free)
"""
import os
import subprocess
import sys
import time

import imageio_ffmpeg
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
FPS = 30
TAIL = 1.0
PORT = 8791
OUT = os.path.join(HERE, "intro_video_v4.mp4")   # 10/25: Carameii music (v3 = Inspired, kept)
WAV = os.path.join(HERE, "sfx", "sfx_track_v4.wav")
QUERY = "capture=1"
if "--short" in sys.argv:   # 10/26 (user): the short cut, played by the page itself (?short=1, intro_html/short_warp.json)
    OUT = os.path.join(HERE, "intro_video_short.mp4")
    WAV = os.path.join(HERE, "sfx", "sfx_track_short.wav")   # from make_intro_short.py
    QUERY = "capture=1&short=1"
    TAIL = 0.0

srv = subprocess.Popen([sys.executable, "-m", "http.server", str(PORT), "--bind", "127.0.0.1", "--directory", HERE],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1.5)
try:
    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
        page.goto(f"http://127.0.0.1:{PORT}/intro_html/intro.html?{QUERY}")
        page.wait_for_function("window.READY === true", timeout=60000)
        dur = page.evaluate("window.DURATION")
        total = dur + TAIL
        n = int(round(total * FPS))
        ff = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                               "-f", "image2pipe", "-framerate", str(FPS), "-i", "-",
                               "-i", WAV, "-af", "apad", "-t", f"{total:.3f}",
                               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "slow",
                               "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", OUT],
                              stdin=subprocess.PIPE)
        t0 = time.time()
        for i in range(n):
            t = min(i / FPS, dur)
            page.evaluate(f"window.seek({t})")
            ff.stdin.write(page.screenshot(type="png"))
            if i % 300 == 0:
                print(f"frame {i}/{n} ({time.time() - t0:.0f}s)", flush=True)
        ff.stdin.close()
        ff.wait()
        b.close()
    print("done:", OUT, f"{os.path.getsize(OUT) / 1e6:.1f} MB")
finally:
    srv.terminate()
