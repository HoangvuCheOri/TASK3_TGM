# webcam_count.py — YOLOv8 webcam (bản đơn giản)
import time, cv2
from ultralytics import YOLO
import torch

def main():
    model_path = r"C:\YOLOTEST\runs\detect\train2\weights\last.pt"
    model = YOLO(model_path)

    # chọn thiết bị
    device = 0 if torch.cuda.is_available() else "cpu"

    # mở webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)   # đổi 0 -> 1/2 nếu cần

    t0, frames = time.time(), 0
    names = model.model.names if hasattr(model, "model") else model.names

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Không đọc được khung hình."); break

        # suy luận 1 khung hình
        res = model(frame, conf=0.5, imgsz=640, device=device, verbose=False)[0]

        # vẽ bbox bằng tiện ích có sẵn
        vis = res.plot()

        # đếm theo lớp (đơn giản)
        counts = {}
        if res.boxes is not None and res.boxes.cls is not None:
            for c in res.boxes.cls.cpu().numpy().astype(int):
                label = names[c] if isinstance(names, list) else names.get(c, str(c))
                counts[label] = counts.get(label, 0) + 1

        # FPS
        frames += 1
        dt = time.time() - t0
        fps = frames / dt if dt > 0 else 0.0

        # overlay thông tin
        y = 24
        cv2.putText(vis, f"FPS: {fps:.1f}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
        y += 28
        if counts:
            cv2.putText(vis, "Count: " + ", ".join(f"{k}:{v}" for k, v in counts.items()),
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)

        cv2.imshow("YOLOv8 Webcam (press q to quit)", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()