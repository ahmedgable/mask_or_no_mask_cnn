import os
import sys
import tempfile
import urllib.request
import cv2
import gradio as gr
import numpy as np
import requests
import tensorflow as tf

# إخفاء تحذيرات TensorFlow المزعجة في الـ Logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# 1. مسارات الأوزان والملفات
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "mask_detector_weights.weights.h5")
FILE_ID = "1JwuFcT__4TsSDdhy6yifntu9Hq_GoI4P"

YUNET_PATH = os.path.join(BASE_DIR, "face_detection_yunet_2023mar.onnx")
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"


def download_file_from_google_drive(file_id, destination):
    """دالة تنزيل الأوزان وتجاوز صفحة الحماية للملفات الكبيرة"""
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={"id": file_id, "confirm": "t"}, stream=True)

    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break

    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)


# تنزيل أوزان نموذج الكمامة إذا لم تكن موجودة
if not os.path.exists(WEIGHTS_PATH):
    print("Downloading weights file from Google Drive...")
    try:
        download_file_from_google_drive(FILE_ID, WEIGHTS_PATH)
        print("Download completed successfully!")
    except Exception as e:
        print(f"Error downloading weights: {e}")
        sys.exit(1)

# 2. بناء نموذج الكشف عن الكمامة
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Input(shape=(128, 128, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

# تحميل الأوزان
model.load_weights(WEIGHTS_PATH)
IMG_SIZE = 128


# 3. فئة كاشف الوجوه باستخدام YuNet
class FaceDetector:
    def __init__(self):
        self.yunet = None
        try:
            if not os.path.exists(YUNET_PATH) or os.path.getsize(YUNET_PATH) < 100_000:
                print("Downloading YuNet face detection model...")
                urllib.request.urlretrieve(YUNET_URL, YUNET_PATH)
            self.yunet = cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320), 0.6, 0.3, 500)
        except Exception as e:
            print(f"Error loading YuNet face detector: {e}")

    def detect(self, bgr):
        if self.yunet is None:
            return []
        h, w = bgr.shape[:2]
        self.yunet.setInputSize((w, h))
        _, det = self.yunet.detect(bgr)
        faces = []
        for d in (det if det is not None else []):
            x, y, fw, fh = map(int, d[:4])
            # ضمان بقاء الإحداثيات داخل حدود الصورة
            x, y = max(0, x), max(0, y)
            fw, fh = min(w - x, fw), min(h - y, fh)
            if fw > 0 and fh > 0:
                faces.append((x, y, fw, fh))
        return faces


detector = FaceDetector()


# 4. معالجة الإطارات
def process_frame(image):
    if image is None:
        return None

    # OpenCV يتعامل مع BGR لذا نحول الصورة من RGB القادمة من Gradio
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    faces = detector.detect(bgr_image)

    output_image = image.copy()

    for x, y, w, h in faces:
        face_crop = output_image[y : y + h, x : x + w]
        if face_crop.size == 0:
            continue

        resized_face = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))
        normalized_face = resized_face / 255.0
        input_tensor = np.expand_dims(normalized_face, axis=0)

        prediction = model.predict(input_tensor, verbose=0)[0][0]

        if prediction > 0.5:
            label = f"Mask: {prediction * 100:.1f}%"
            color = (0, 255, 0)
        else:
            label = f"No Mask: {(1 - prediction) * 100:.1f}%"
            color = (255, 0, 0)

        cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 3)
        cv2.putText(
            output_image,
            label,
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

    return output_image


# 5. واجهة Gradio
demo = gr.Interface(
    fn=process_frame,
    inputs=gr.Image(
        sources=["webcam"],
        type="numpy",
        streaming=True,
        label="الكاميرا المباشرة",
    ),
    outputs=gr.Image(type="numpy", label="التتبع اللحظي للوجه"),
    title="😷 Real-Time Face Mask Detector with Bounding Box",
    live=True,
    theme="soft",
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
