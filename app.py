import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(
    page_title="Meat Quality Classification",
    page_icon="\U0001F969",
    layout="centered"
)

st.title("Sistem Klasifikasi Kualitas Daging")
st.write("Upload gambar daging, lalu sistem akan memprediksi kualitas daging menggunakan model TensorFlow Lite.")

IMG_SIZE = (128, 128)

@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

@st.cache_data
def load_labels():
    with open("labels.txt", "r") as f:
        return [line.strip() for line in f.readlines()]

interpreter = load_tflite_model()
labels = load_labels()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

uploaded_file = st.file_uploader(
    "Upload gambar daging",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diupload", use_container_width=True)

    img = image.resize(IMG_SIZE)

    # PENTING: JANGAN dibagi 255. Model dilatih menggunakan EfficientNetB0,
    # yang sudah memiliki layer normalisasi built-in dan mengharapkan input
    # piksel mentah skala 0-255. Membagi dengan 255 di sini akan membuat
    # prediksi menjadi salah (model akan selalu condong ke satu kelas saja),
    # sama seperti bug yang menyebabkan model versi awal selalu memprediksi
    # "Fresh".
    img_array = np.array(img).astype(input_details[0]["dtype"])
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]["index"])[0]

    predicted_index = int(np.argmax(prediction))
    predicted_class = labels[predicted_index]
    confidence = float(np.max(prediction) * 100)

    st.subheader("Hasil Prediksi")
    st.write(f"**Kelas:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")
    st.progress(confidence / 100)

    with st.expander("Lihat detail confidence tiap kelas"):
        for label, score in zip(labels, prediction):
            st.write(f"{label}: {float(score) * 100:.2f}%")
