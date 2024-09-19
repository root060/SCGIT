"""
TODO : streamlit
TODO : MNIST model
"""
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import ImageOps, Image
from streamlit_drawable_canvas import st_canvas

model = tf.keras.models.load_model('cnn-mnist-model.h5')

def preprocess_image(image):
    image = image.resize((28, 28))
    image = np.array(image)
    image = image.reshape(1, 28, 28, 1).astype('float32') / 255
    return image

st.title('손글씨 숫자 인식기 - 그려보기')
st.write('캔버스에 숫자를 그린 후, 예측 버튼을 누르세요.')

canvas_result = st_canvas(
    stroke_width=10,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=150,
    width=150,
    drawing_mode="freedraw",
    key="canvas"
)

if canvas_result.image_data is not None:
    img = Image.fromarray(np.uint8(canvas_result.image_data))
    processed_image = preprocess_image(img)

    prediction = model.predict(processed_image)
    predicted_digit = np.argmax(prediction, axis=1)

    st.write(f"예측된 숫자는: {predicted_digit[0]}")