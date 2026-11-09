import time, subprocess
from lerobot.robots.so_follower import SOFollowerRobotConfig, SOFollower
cfg = SOFollowerRobotConfig(port="/dev/tty.usbmodem5C821074811", id="my_follower")
r = SOFollower(cfg)
r.bus.connect()
r.bus.disable_torque("gripper")
print("torque_enable(gripper) =", r.bus.read("Torque_Enable", "gripper", normalize=False), flush=True)
subprocess.Popen(["say", "測定を開始します。グリッパーを最後まで閉じて、次に最後まで開いてください"])
lo, hi = 10**9, -10**9
t0 = time.time()
while time.time() - t0 < 75:
    p = r.bus.read("Present_Position", "gripper", normalize=False)
    w = r.bus.read("Present_Position", "wrist_roll", normalize=False)
    lo, hi = min(lo, p), max(hi, p)
    print(f"{time.time()-t0:5.1f}s grip={p} wrist_roll={w} min={lo} max={hi}", flush=True)
    time.sleep(0.2)
subprocess.run(["say", "測定終了"])
print(f"RESULT min={lo} max={hi}", flush=True)
r.bus.port_handler.closePort()
