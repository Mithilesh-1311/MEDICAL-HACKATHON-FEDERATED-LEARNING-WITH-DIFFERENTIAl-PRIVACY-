import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

st.title("Federated Medical Diagnosis - Global Model Demo")
st.write("Upload an X-ray image to classify Normal vs Pneumonia.")


def build_model(weights):
    IMG_SHAPE = (100,100,3)
    base_model = tf.keras.applications.EfficientNetV2S(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=IMG_SHAPE,
        pooling=None,
        classes=2,
        classifier_activation='softmax',
        include_preprocessing=True
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(100,100,3))
    x = base_model(inputs, training=False)  
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    if type(weights) is list:
        model.set_weights(weights)
    else:
        model.load_weights(weights)
    
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001),
               loss = tf.keras.losses.CategoricalCrossentropy(),
                metrics = ['accuracy',tf.keras.metrics.Precision(name = 'pn'),tf.keras.metrics.Recall(name = 'rc'),tf.keras.metrics.F1Score(name='f1'),tf.keras.metrics.FalseNegatives(name='fn'),tf.keras.metrics.FalsePositives(name='fp'),tf.keras.metrics.TrueNegatives(name='tn'),tf.keras.metrics.TruePositives(name='tp')],
    )

    return model


@st.cache_resource
def load_model():
    return build_model(Path(__file__).resolve().parent / "HOSPITAL" / "global_model.weights.h5")

model = load_model()


uploaded_img = st.file_uploader("Upload a Chest X-ray Image", type=["png", "jpg", "jpeg"])

if uploaded_img:
    img = Image.open(uploaded_img).convert("RGB")
    img_resized = img.resize((100,100))

    st.image(img_resized, caption="Uploaded Image", width=200)

    x = np.array(img_resized)/99.0
    x = np.expand_dims(x, axis=0)

    pred = model.predict(x)
    class_idx = np.argmax(pred)
    confidence = pred[0][class_idx]

    label = "Pneumonia" if class_idx == 1 else "Normal"

    st.subheader(f"Prediction: **{label}**")
    st.write(f"Confidence: **{confidence*100:.2f}%**")
