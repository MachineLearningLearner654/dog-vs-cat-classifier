import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load your model
model = load_model('dog_cat_finetuned_model.h5')


# Class names
class_names = ['Cat', 'Dog']

# Streamlit UI
st.title("🐶🐱 Dog vs Cat Classifier")
st.write("Upload an image and I'll tell you if it's a dog or a cat!")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image.', use_column_width=True)

    # Preprocess image
    img = img.resize((224, 224))  # Resize to match model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalization if model was trained on this

    # Predict
    prediction = model.predict(img_array)[0][0]
    result = "Dog" if prediction > 0.5 else "Cat"

    st.write(f"### Prediction: {result}")
    st.write(f"Confidence: {prediction:.4f}")
