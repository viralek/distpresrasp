#!/usr/bin/env python3
"""
YOLO Person Tracker & Pixelator (video‑only)
Raspberry Pi 5 • Camera Module 3 (Wide) • Python 3.11

✓ picamera2 nahrádza OpenCV VideoCapture/GStreamer
✓ audio časť je deaktivovaná (a možno ju neskôr vrátiť)
✓ vhodné pre monitor / projektor (HDMI duplikovať alebo rozšíriť)
"""

from pathlib import Path
import sys, signal
import cv2, numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO

# ──────────────────────── KONFIG ────────────────────────
MODEL_PATH = "yolov8n-seg.pt"       # YOLO v8 segmentation model

# Výstupný náhľad na monitor/projektor
OUT_W, OUT_H, FPS = 1280, 720, 30

STATUSBAR    = True    # HUD so súradnicami/osobami
MPACTIVE     = True    # Pixelácia detegovaných osôb
TRANSPARENCY = True    # Stmavenie obrazov mimo masiek

# Audio – vypnuté, kód ostáva pre budúcnosť
audioactive  = False
# ────────────────────────────────────────────────────────

def percent(val, max_val):
    return (val / max_val) * 100.0 if max_val else 0.0

def pixelate_roi(roi: np.ndarray, block: int) -> np.ndarray:
    h, w = roi.shape[:2]
    if h < 2 or w < 2 or block < 1:
        return roi
    small = cv2.resize(roi, (max(1, w // block), max(1, h // block)))
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def init_camera() -> Picamera2:
    picam = Picamera2()
    cfg = picam.create_video_configuration(
        main={"size": (OUT_W, OUT_H), "format": "RGB888"},
        controls={"FrameRate": FPS})
    picam.configure(cfg)
    picam.start()
    return picam

def main() -> None:
    if not Path(MODEL_PATH).exists():
        sys.exit(f"[ERROR] Model {MODEL_PATH} chýba")

    print("[INFO] Načítavam YOLO…")
    model = YOLO(MODEL_PATH)

    print("[INFO] Spúšťam kameru…")
    picam = init_camera()
    fw, fh = OUT_W, OUT_H
    scene_pix = fw * fh

    win = "Person Tracker (q/ESC exit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, fw // 2, fh // 2)

    def cleanup(*_):
        print("\n[INFO] Končím…")
        picam.close()
        cv2.destroyAllWindows()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)

    while True:
        frame_rgb = picam.capture_array()            # RGB888
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        res = model.predict(frame, classes=[0], verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy() if hasattr(res.boxes, "xyxy") else []
        masks = res.masks.data.cpu().numpy() if getattr(res, "masks", None) is not None else None

        out = np.zeros_like(frame) if TRANSPARENCY else frame.copy()
        persons = []

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cx = percent((x1 + x2) / 2, fw)
            cy = percent((y1 + y2) / 2, fh)
            area_pct = percent((x2 - x1) * (y2 - y1), scene_pix)
            persons.append((cx, cy, area_pct))

            mask = masks[i].astype(bool) if masks is not None else \
                cv2.rectangle(np.zeros((fh, fw), bool), (x1, y1), (x2, y2), True, -1)
            roi = frame[y1:y2, x1:x2]
            roi_px = pixelate_roi(roi, max(4, int(area_pct / 5))) if MPACTIVE else roi
            dst = out[y1:y2, x1:x2]
            dst[mask[y1:y2, x1:x2]] = roi_px[mask[y1:y2, x1:x2]]
            out[y1:y2, x1:x2] = dst

        if STATUSBAR:
            bar_h = 28
            cv2.rectangle(out, (0, fh - bar_h), (fw, fh), (0, 0, 0), -1)
            text = f"{len(persons)} ppl: " + \
                   ", ".join(f"{int(cx)}%,{int(cy)}%" for cx, cy, _ in persons)
            cv2.putText(out, text, (8, fh - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1)

        cv2.imshow(win, out)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            cleanup()

if __name__ == "__main__":
    main()
