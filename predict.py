"""Facial emotion recognition from a photo, using the exported ONNX model.

Pipeline: image -> detect faces (OpenCV) -> crop -> grayscale -> resize 224 -> normalize
          -> EfficientNet-B0 (ONNX) -> emotion + confidence

Usage:
    python predict.py photo.jpg
    python predict.py photo.jpg --save result.jpg
    python predict.py photo.jpg --model models/fer_efficientnet_b0_fp32.onnx
"""
import argparse

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
IMG_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)  # ImageNet mean
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)   # ImageNet std


def detect_faces(gray):
    """Return face boxes (x, y, w, h). Uses OpenCV's Haar cascade (OpenCV 4.x)."""
    if not hasattr(cv2, "CascadeClassifier"):
        return []
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    return [tuple(int(v) for v in f) for f in faces]


def preprocess(gray_face):
    """Same preprocessing as the validation/test images during training."""
    face = Image.fromarray(gray_face).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    x = np.asarray(face, dtype=np.float32) / 255.0
    x = np.stack([x, x, x])                       # 1 gray channel -> 3 channels
    return ((x - MEAN) / STD)[None].astype(np.float32)   # shape [1, 3, 224, 224]


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def main():
    parser = argparse.ArgumentParser(description="Predict facial emotions in an image.")
    parser.add_argument("image", help="path to an image file")
    parser.add_argument("--model", default="models/fer_efficientnet_b0_fp16.onnx", help="ONNX model path")
    parser.add_argument("--save", help="save an annotated copy of the image to this path")
    args = parser.parse_args()

    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    rgb = np.array(Image.open(args.image).convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    boxes = detect_faces(gray)
    if not boxes:                                  # no face found -> use the whole image
        print("No face detected - using the whole image.")
        boxes = [(0, 0, gray.shape[1], gray.shape[0])]

    for n, (x, y, w, h) in enumerate(boxes, 1):
        m = int(0.1 * w)                           # small margin around the face
        x0, y0 = max(x - m, 0), max(y - m, 0)
        x1, y1 = min(x + w + m, gray.shape[1]), min(y + h + m, gray.shape[0])
        logits = session.run(None, {"image": preprocess(gray[y0:y1, x0:x1])})[0][0]
        probs = softmax(logits.astype(np.float32))
        top = np.argsort(-probs)[:3]
        print(f"Face {n}: " + " | ".join(f"{CLASS_NAMES[i]} {100 * probs[i]:.1f}%" for i in top))
        if args.save:
            label = f"{CLASS_NAMES[top[0]]} {100 * probs[top[0]]:.0f}%"
            cv2.rectangle(rgb, (x0, y0), (x1, y1), (0, 200, 0), 2)
            cv2.putText(rgb, label, (x0, max(y0 - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

    if args.save:
        Image.fromarray(rgb).save(args.save)
        print("Saved:", args.save)


if __name__ == "__main__":
    main()
