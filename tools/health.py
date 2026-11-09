import time
from lerobot.robots.so_follower import SOFollowerRobotConfig, SOFollower
r = SOFollower(SOFollowerRobotConfig(port="/dev/tty.usbmodem5C821074811", id="my_follower"))
r.bus.connect()
ok = err = 0; last = None
t0 = time.time()
while time.time() - t0 < 10:
    try:
        last = r.bus.sync_read("Present_Position", normalize=False); ok += 1
    except Exception as e:
        err += 1
    time.sleep(0.02)
print("reads ok", ok, "errors", err)
print("positions", last)
for m in r.bus.motors:
    try: print(m, "voltage", r.bus.read("Present_Voltage", m, normalize=False))
    except Exception as e: print(m, "voltage read failed", type(e).__name__)
r.bus.port_handler.closePort()
