import os
import urllib.request
import cv2
import gradio as gr
import numpy as np
import tensorflow as tf

# 1. تحديد مسار ملف الأوزان ورابط التحميل المباشر من Google Drive
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "mask_detector_weights.weights.h5")
DRIVE_FILE_ID = "1JwuFcT__4TsSDdhy6yifntu9Hq_GoI4P"
WEIGHTS_URL = f"https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}"

# تنزيل ملف الأوزان تلقائياً إذا لم يكن موجوداً داخل السيرفر
if not os.path.exists(WEIGHTS_PATH):
    print("Downloading weights file from Google Drive...")
    urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_PATH)
    print("Download complete!")

# 2. بناء شبكة الـ CNN
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

# 3. تحميل الأوزان إلى النموذج
model.load_weights(WEIGHTS_PATH)

IMG_SIZE = 128

# 4. تحميل كاشف الوجوه المدمج من OpenCV
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# 5. دالة معالجة إطار الفيديو المباشر
def process_frame(image):
    if image is None:
        return None

    # تحويل الصورة إلى درجات الرمادي لكشف الوجه
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # كشف مكان الوجه في الإطار
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    output_image = image.copy()

    for x, y, w, h in faces:
        face_crop = output_image[y : y + h, x : x + w]

        # تجهيز منطقة الوجه للتنبؤ
        resized_face = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))
        normalized_face = resized_face / 255.0
        input_tensor = np.expand_dims(normalized_face, axis=0)

        # التنبؤ بواسطة النموذج
        prediction = model.predict(input_tensor, verbose=0)[0][0]

        # تحديد النتيجة واللون (RGB ليتناسب مع Gradio)
        if prediction > 0.5:
            label = f"Mask: {prediction*100:.1f}%"
            color = (0, 255, 0)  # أخضر
        else:
            label = f"No Mask: {(1-prediction)*100:.1f}%"
            color = (255, 0, 0)  # أحمر

        # رسم المربع حول الوجه وتدوين النسبة
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


# 6. إعداد واجهة Gradio
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
    theme="soft",
    live=True,
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)