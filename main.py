import os
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import ImageOps, Image
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# code line 11 to 16 is from rahulsrma26
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
if not os.path.isdir(MODEL_DIR):
    os.system('runipy train.ipynb')

model = ts.keras.models.load_model('model')
# st.markdown('<style>body{color: White; background-color: DarkSlateGrey}</style>', unsafe_allow_html=True)

# 이미지 전처리 함수
def preprocess_image(image):
    image = image.resize((28, 28))
    image = np.array(image)
    image = image.reshape(1, 28, 28, 1).astype('float32') / 255
    return image

# Streamlit 앱 시작
st.title('손글씨 숫자 인식기 - 그려보기')
st.write('캔버스에 숫자를 그린 후, 예측 버튼을 누르세요.')

# 그리기 모드
mode = st.checkbox("그리기 (혹은 지우기)?", True)

# 그리기 캔버스
canvas_result = st_canvas(
    stroke_width=10,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=150,
    width=150,
    drawing_mode="freedraw" if mode else "transform",
    key="canvas"
)

# 사용자가 그림을 그린 경우 예측 수행
if canvas_result.image_data is not None:
    img = Image.fromarray(np.uint8(canvas_result.image_data))
    processed_image = preprocess_image(img)

    prediction = model.predict(processed_image)
    predicted_digit = np.argmax(prediction, axis=1)

    st.write(f"예측된 숫자는: {predicted_digit[0]}")
