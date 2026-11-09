import time, subprocess
from lerobot.robots.so_follower import SOFollowerRobotConfig, SOFollower
r = SOFollower(SOFollowerRobotConfig(port="/dev/tty.usbmodem5C821074811", id="my_follower"))
r.bus.connect(); r.bus.disable_torque()
steps = [("肩の回転",8),("肩の上下",8),("ひじ",8),("手首の曲げ",8),("手首の回転",8),("グリッパー",8)]
subprocess.run(["say","動作テストを始めます"])
for name, dur in steps:
    subprocess.Popen(["say", name])
    ok = err = 0; lo = {}; hi = {}
    t0 = time.time()
    while time.time() - t0 < dur:
        try:
            p = r.bus.sync_read("Present_Position", normalize=False); ok += 1
            for m, v in p.items(): lo[m] = min(lo.get(m, v), v); hi[m] = max(hi.get(m, v), v)
        except Exception as e:
            err += 1
        time.sleep(0.02)
    moved = {m: hi[m]-lo[m] for m in lo if hi[m]-lo[m] > 30}
    print(f"{name}: ok={ok} err={err} moved={moved}", flush=True)
subprocess.run(["say","テスト終了"])
r.bus.port_handler.closePort()
