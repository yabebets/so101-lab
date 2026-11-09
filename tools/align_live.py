"""Live camera alignment helper: overlays the recording-time reference frame and estimates the offset."""
import sys, time, cv2, numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("local/so101_block_place")
ref = np.ascontiguousarray((ds[int(ds.meta.episodes[int(sys.argv[3]) if len(sys.argv) > 3 else 5]["dataset_from_index"])]["observation.images.front"]
                            .permute(1, 2, 0).numpy() * 255).astype(np.uint8)[:, :, ::-1])
ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
ref_edges = cv2.Canny(ref_gray, 60, 160)
orb = cv2.ORB_create(1500)
kr, dr = orb.detectAndCompute(ref_gray, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

cap = cv2.VideoCapture(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
cap.set(3, 640); cap.set(4, 480)
t_end = time.time() + 600
last_print = 0
while time.time() < t_end:
    ok, f = cap.read()
    if not ok:
        time.sleep(0.05); continue
    f = cv2.resize(f, (640, 480))
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    kl, dl = orb.detectAndCompute(g, None)
    msg, ok_fit = "no match", False
    if dl is not None and len(kl) > 20:
        m = sorted(bf.match(dr, dl), key=lambda x: x.distance)[:300]
        if len(m) >= 12:
            A, inl = cv2.estimateAffinePartial2D(np.float32([kr[x.queryIdx].pt for x in m]),
                                                 np.float32([kl[x.trainIdx].pt for x in m]),
                                                 ransacReprojThreshold=4.0)
            if A is not None and inl.sum() >= 12:
                s = float(np.hypot(A[0, 0], A[1, 0]))
                c = A @ np.array([320, 240, 1.0])
                dx, dy = (c[0] - 320) / 640 * 100, (c[1] - 240) / 480 * 100
                hints = []
                if dx > 3: hints.append("PAN RIGHT")
                if dx < -3: hints.append("PAN LEFT")
                if dy > 3: hints.append("TILT DOWN")
                if dy < -3: hints.append("TILT UP")
                if s > 1.05: hints.append("MOVE BACK")
                if s < 0.95: hints.append("MOVE CLOSER")
                ok_fit = not hints
                msg = f"dx={dx:+.0f}% dy={dy:+.0f}% scale={s:.2f}  " + (" / ".join(hints) if hints else "OK!")
                if time.time() - last_print > 1:
                    print(msg, flush=True); last_print = time.time()
    view = f.copy()
    view[ref_edges > 0] = (0, 0, 255)
    blend = cv2.addWeighted(ref, 0.5, f, 0.5, 0)
    out = np.hstack([view, blend])
    cv2.rectangle(out, (0, 0), (1280, 40), (0, 0, 0), -1)
    cv2.putText(out, msg, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if ok_fit else (0, 200, 255), 2)
    cv2.putText(out, "red lines = recording view   |   right: 50% overlay   |   press q to close", (10, 470),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow("camera alignment", out)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cv2.imwrite(sys.argv[2] if len(sys.argv) > 2 else "align_final.jpg", out)
cap.release(); cv2.destroyAllWindows()
