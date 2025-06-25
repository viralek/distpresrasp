#!/usr/bin/env python3
"""
YOLO Person Tracker + Pixelácia (bez audia)
Raspberry Pi 5  •  Camera Module 3 Wide  •  Python 3.11
"""

from pathlib import Path
import sys, signal, cv2, numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO

# ─────────── KONFIG ───────────
MODEL_PATH  = "yolov8n-seg.pt"
OUT_W, OUT_H, FPS = 1280, 720, 30     # náhľad na monitor/projektor

STATUSBAR    = True                   # textový HUD
MPACTIVE     = True                   # pixelácia osôb
TRANSPARENCY = True                   # stmavenie pozadia mimo masiek
# ──────────────────────────────

# ───── Pomocné funkcie ─────
def percent(val, max_val): return (val / max_val) * 100 if max_val else 0

def pixelate(img, block=20):
    h, w = img.shape[:2]
    if h < 2 or w < 2: return img
    tmp = cv2.resize(img, (w//block, h//block), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(tmp, (w, h), interpolation=cv2.INTER_NEAREST)
# ───────────────────────────

def main():
    # ––– kontrola modelu –––
    if not Path(MODEL_PATH).exists():
        sys.exit(f"[ERROR] Model {MODEL_PATH} neexistuje.")
    print("[INFO] Načítavam YOLO model…")
    model = YOLO(MODEL_PATH)

    # ––– inicializácia kamery cez picamera2 –––
    print("[INFO] Inicializujem kameru…")
    picam = Picamera2()
    cfg = picam.create_video_configuration(
        main={"size": (OUT_W, OUT_H), "format": "RGB888"},
        controls={"FrameRate": FPS})
    picam.configure(cfg)
    picam.start()

    win = "Person Tracker  (q/ESC exit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, OUT_W//1.3, OUT_H//1.3)

    # ––– SIGINT handler –––
    def cleanup(*_):
        print("\n[INFO] Končím…")
        picam.close()
        cv2.destroyAllWindows()
        sys.exit(0)
    signal.signal(signal.SIGINT, cleanup)

    fw, fh = OUT_W, OUT_H
    scene_vol = fw * fh

    while True:
        frame = picam.capture_array()               # RGB
        bgr   = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # YOLO detekcia len osôb
        res = model.predict(bgr, classes=[0], verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy() if hasattr(res.boxes, "xyxy") else []
        masks = res.masks.data.cpu().numpy() if getattr(res, "masks", None) is not None else None

        out = np.zeros_like(bgr) if TRANSPARENCY else bgr.copy()
        persons = []

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            x1,y1,x2,y2 = map(int, (x1,y1,x2,y2))
            cx = percent((x1+x2)/2, fw); cy = percent((y1+y2)/2, fh)
            area_pct = percent((x2-x1)*(y2-y1), scene_vol)
            persons.append((cx, cy, area_pct))

            mask = masks[i].astype(bool) if masks is not None else \
                   cv2.rectangle(np.zeros((fh,fw), bool), (x1,y1), (x2,y2), True, -1)
            roi = bgr[y1:y2, x1:x2]
            roi_proc = pixelate(roi, max(4, int(area_pct/5))) if MPACTIVE else roi
            dst = out[y1:y2, x1:x2]
            dst[mask[y1:y2, x1:x2]] = roi_proc[mask[y1:y2, x1:x2]]
            out[y1:y2, x1:x2] = dst

        # HUD
        if STATUSBAR:
            bar_h = 28
            cv2.rectangle(out, (0, fh-bar_h), (fw, fh), (0,0,0), -1)
            txt = f"{len(persons)} ppl: " + ", ".join(f"{int(cx)}%,{int(cy)}%" for cx,cy,_ in persons)
            cv2.putText(out, txt, (8, fh-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        cv2.imshow(win, out)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            cleanup()

if __name__ == "__main__":
    main()
