import os
import sys

# إخفاء تحذيرات TensorFlow المزعجة في الـ Logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import gradio as gr
import numpy as np
import requests
import tensorflow as tf

# 1. تحديد مسار الأوزان ورابط التنزيل
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "mask_detector_weights.weights.h5")
FILE_ID = "1JwuFcT__4TsSDdhy6yifntu9Hq_GoI4P"


def download_file_from_google_drive(file_id, destination):
    """دالة تنزيل الأوزان وتجاوز صفحة الحماية للملفات الكبيرة"""
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(
        URL, params={"id": file_id, "confirm": "t"}, stream=True
    )

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



if not os.path.exists(WEIGHTS_PATH):
    print("Downloading weights file from Google Drive...")
    try:
        download_file_from_google_drive(FILE_ID, WEIGHTS_PATH)
        print("Download completed successfully!")
    except Exception as e:
        print(f"Error downloading weights: {e}")
        sys.exit(1)


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

# 4. تحميل الأوزان
model.load_weights(WEIGHTS_PATH)

IMG_SIZE = 128
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# 5. معالجة الإطارات
def process_frame(image):
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    output_image = image.copy()

    for x, y, w, h in faces:
        face_crop = output_image[y : y + h, x : x + w]
        resized_face = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))
        normalized_face = resized_face / 255.0
        input_tensor = np.expand_dims(normalized_face, axis=0)

        prediction = model.predict(input_tensor, verbose=0)[0][0]

        if prediction > 0.5:
            label = f"Mask: {prediction*100:.1f}%"
            color = (0, 255, 0)
        else:
            label = f"No Mask: {(1-prediction)*100:.1f}%"
            color = (255, 0, 0)

        cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 3)
        cv2.putText(
            output_image,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

    return output_image


# 6. واجهة Gradio
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
)

# 2. إضافة theme هنا داخل launch
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, theme="soft")