"""lerobot-record with a relaxed camera frame-age limit (Continuity Camera occasionally stalls > 500 ms)."""
import functools, sys
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera

_orig = OpenCVCamera.read_latest

@functools.wraps(_orig)
def read_latest(self, max_age_ms: int = 2000):
    return _orig(self, max_age_ms=max(max_age_ms, 2000))

OpenCVCamera.read_latest = read_latest

from lerobot.scripts.lerobot_record import main
sys.exit(main())
