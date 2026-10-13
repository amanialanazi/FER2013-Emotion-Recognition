"""Live facial emotion recognition from a webcam (real-time video stream).

Uses the best model (Full Fine-Tuning) exported to ONNX FP16 — no PyTorch, no GPU needed.

Pipeline for every video frame:
    frame -> detect faces (OpenCV Haar cascade) -> crop each face -> grayscale -> resize 224
          -> normalize -> EfficientNet-B0 (ONNX) -> emotion + confidence -> draw on the frame

Usage:
    pip install onnxruntime "opencv-python<5" numpy
    python live_webcam.py                 # laptop webcam (camera 0)
    python live_webcam.py --camera 1      # another camera
    python live_webcam.py --source video.mp4 --save out.mp4   # run on a video file
Press  q  to quit.
"""
import argparse
import time

import cv2
import numpy as np
import onnxruntime as ort

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
COLORS = {  # BGR colour per emotion for the box and label
    "angry": (60, 60, 230), "disgust": (60, 160, 60), "fear": (180, 80, 160), "happy": (40, 200, 240),
    "neutral": (200, 200, 200), "sad": (220, 140, 60), "surprise": (40, 160, 255),
}
IMG_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)  # ImageNet mean
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)   # ImageNet std


def preprocess(gray_face):
    """Same preprocessing as the validation/test images during training."""
    face = cv2.resize(gray_face, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    x = face.astype(np.float32) / 255.0
    x = np.stack([x, x, x])                                 # 1 gray channel -> 3 channels
    return ((x - MEAN) / STD)[None].astype(np.float32)      # shape [1, 3, 224, 224]


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def main():
    ap = argparse.ArgumentParser(description="Live facial emotion recognition.")
    ap.add_argument("--model", default="models/fer_efficientnet_b0_fp16.onnx")
    ap.add_argument("--camera", type=int, default=0, help="webcam index")
    ap.add_argument("--source", help="video file instead of the webcam")
    ap.add_argument("--save", help="save the annotated video to this .mp4 path")
    ap.add_argument("--smooth", type=float, default=0.6,
                    help="0 = no smoothing, 0.9 = very smooth (averages predictions over frames)")
    args = ap.parse_args()

    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(args.source if args.source else args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera / video. Try --camera 1, or check camera permissions.")

    writer, smoothed, fps, t_prev = None, None, 0.0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60))

        for (x, y, w, h) in faces:
            m = int(0.1 * w)                                # small margin around the face
            x0, y0 = max(x - m, 0), max(y - m, 0)
            x1, y1 = min(x + w + m, gray.shape[1]), min(y + h + m, gray.shape[0])
            probs = softmax(session.run(None, {"image": preprocess(gray[y0:y1, x0:x1])})[0][0].astype(np.float32))
            if len(faces) == 1:                             # smooth over frames to reduce flicker (one face)
                smoothed = probs if smoothed is None else args.smooth * smoothed + (1 - args.smooth) * probs
                probs = smoothed
            label = CLASS_NAMES[int(probs.argmax())]
            color = COLORS[label]
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
            cv2.putText(frame, f"{label} {100 * probs.max():.0f}%", (x0, max(y0 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            for rank, i in enumerate(np.argsort(-probs)[:3]):   # top-3 bars under the face
                yy = y1 + 18 + 18 * rank
                cv2.rectangle(frame, (x0, yy - 10), (x0 + int(120 * probs[i]), yy), COLORS[CLASS_NAMES[i]], -1)
                cv2.putText(frame, f"{CLASS_NAMES[i]} {100 * probs[i]:.0f}%", (x0 + 125, yy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
        t_prev = now
        cv2.putText(frame, f"FPS {fps:.1f} | faces {len(faces)} | q = quit", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if args.save:
            if writer is None:
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), 20,
                                         (frame.shape[1], frame.shape[0]))
            writer.write(frame)
        if not args.source or not args.save:                # show a window (skip when batch-processing a file)
            cv2.imshow("Facial Emotion Recognition - live", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
        print("Saved:", args.save)
    if not args.source or not args.save:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
