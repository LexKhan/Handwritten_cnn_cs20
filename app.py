import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image

from preproces import preprocess_image

# Page config
st.set_page_config(
    page_title="Handwritten Character Recognition",
    layout="centered"
)

st.title("📝 Handwritten Character Recognition")
st.write("Upload a **black stroke on white background** image")

# Load model & labels
@st.cache_resource
def load_assets():
    model = load_model("model/handwritten_cnn_model (2).keras")
    class_names = np.load("model/class_names (2).npy", allow_pickle=True)
    return model, class_names

model, class_names = load_assets()
st.success(f"✅ Model loaded ({len(class_names)} classes)")

# Upload image
uploaded_file = st.file_uploader(
    "📂 Upload an image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    img_pil = Image.open(uploaded_file).convert("RGB")
    st.image(img_pil, caption="Uploaded Image", width=300)

    img_array = preprocess_image(img_pil)

    if img_array is None:
        st.warning("⚠️ No visible strokes detected.")
    else:
        # Predict
        predictions = model.predict(img_array)[0]

        predicted_index = int(np.argmax(predictions))
        predicted_label = class_names[predicted_index]
        confidence = predictions[predicted_index] * 100

        st.subheader("🧠 Prediction")

        if confidence < 50:
            st.warning("Low confidence — unclear input")
        else:
            st.success(f"**{predicted_label}** ({confidence:.2f}%)")

        # Top 3 predictions
        st.subheader("📊 Top 3 Predictions")
        top_3 = predictions.argsort()[-3:][::-1]

        for idx in top_3:
            st.write(f"**{class_names[idx]}** — {predictions[idx]*100:.2f}%")
            st.progress(float(predictions[idx]))
