import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import ImageOps, Image
from streamlit_drawable_canvas import st_canvas

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# MNIST 데이터셋 로드
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# CNN 모델 정의
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# 모델 컴파일 및 학습
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5)

# 학습된 모델 저장 (SavedModel 형식)
model.save('saved_model_format')


def preprocess_image(image):
    image = image.resize((28, 28))
    image = np.array(image)
    image = image.reshape(1, 28, 28, 1).astype('float32') / 255
    return image

st.title('손글씨 숫자 인식기 - 그려보기')
st.write('캔버스에 숫자를 그린 후, 예측 버튼을 누르세요.')

mode = st.checkbox("그리기 (혹은 지우기)?", True)
canvas_result = st_canvas(
    stroke_width=10,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=150,
    width=150,
    drawing_mode="freedraw" if mode else "transform",
    key="canvas"
)
'''
if canvas_result.image_data is not None:
    img = Image.fromarray(np.uint8(canvas_result.image_data))
    processed_image = preprocess_image(img)

    prediction = model.predict(processed_image)
    predicted_digit = np.argmax(prediction, axis=1)

    st.write(f"예측된 숫자는: {predicted_digit[0]}")
'''
