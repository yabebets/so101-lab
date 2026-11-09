"""Follower calibration; steps advance when signal files appear (instead of ENTER)."""
import os, sys, time
from lerobot.motors import MotorCalibration
from lerobot.motors.feetech import OperatingMode
from lerobot.robots.so_follower import SOFollowerRobotConfig, SOFollower

S = sys.argv[1]
def wait_for(name):
    while not os.path.exists(os.path.join(S, name)):
        time.sleep(0.1)

r = SOFollower(SOFollowerRobotConfig(port="/dev/tty.usbmodem5C821074811", id="my_follower"))
r.bus.connect()
r.bus.disable_torque()
for m in r.bus.motors:
    r.bus.write("Operating_Mode", m, OperatingMode.POSITION.value)
print("READY_STEP1", flush=True)
wait_for("step1")
homing = r.bus.set_half_turn_homings()
print("homing", homing, flush=True)

motors = [m for m in r.bus.motors if m != "wrist_roll"]
pos = r.bus.sync_read("Present_Position", motors, normalize=False)
mins, maxes = dict(pos), dict(pos)
print("RECORDING_STEP2", flush=True)
errs = 0
while not os.path.exists(os.path.join(S, "step2")):
    try:
        pos = r.bus.sync_read("Present_Position", motors, normalize=False, num_retry=2)
    except ConnectionError:
        errs += 1; time.sleep(0.05); continue
    for m in motors:
        mins[m] = min(mins[m], pos[m]); maxes[m] = max(maxes[m], pos[m])
    time.sleep(0.02)
mins["wrist_roll"], maxes["wrist_roll"] = 0, 4095
print("read errors", errs, flush=True)
print("ranges", {m: (mins[m], maxes[m], maxes[m] - mins[m]) for m in r.bus.motors}, flush=True)
bad = [m for m in motors if maxes[m] - mins[m] < 300]
if bad:
    print("NOT SAVED; too small range:", bad, flush=True)
else:
    r.calibration = {m: MotorCalibration(id=mm.id, drive_mode=0, homing_offset=homing[m],
                                         range_min=mins[m], range_max=maxes[m]) for m, mm in r.bus.motors.items()}
    r.bus.write_calibration(r.calibration)
    r._save_calibration()
    print("SAVED", r.calibration_fpath, flush=True)
r.bus.port_handler.closePort()
