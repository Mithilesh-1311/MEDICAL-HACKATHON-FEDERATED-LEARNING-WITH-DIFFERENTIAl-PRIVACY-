import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

st.set_page_config(page_title="Chest X-ray | Federated Medical", layout="wide")
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #f7f8f4; }
[data-testid="stHeader"] { background: #f7f8f4; }
.block-container { max-width: 1160px; padding-top: 3rem; padding-bottom: 2rem; }
h1, h2, h3, p { color: #263c35; }
h1 { font-family: Georgia, 'Times New Roman', serif !important; font-weight: 400 !important;
     font-size: clamp(2.5rem, 5vw, 4.2rem) !important; letter-spacing: -.045em;
     line-height: 1.08 !important; max-width: 740px; padding-bottom: .8rem; }
h2 { font-size: 1.3rem !important; letter-spacing: -.025em; }
h3 { font-size: 1.1rem !important; }
.masthead { display: flex; justify-content: space-between; align-items: center;
    gap: 1rem; padding-bottom: 1.3rem; border-bottom: 1px solid #dce2d9;
    margin-bottom: 2.8rem; color: #263c35; }
.brand { display: flex; align-items: center; gap: .65rem; font-weight: 600; }
.brand-mark { background: #285747; color: #fff; width: 30px; height: 30px;
    display: grid; place-items: center; border-radius: 6px; font-size: 24px; }
.meta, .eyebrow, .step-label { font-size: .75rem; letter-spacing: .08em; }
.meta { color: #56675f; }
.eyebrow { color: #456454; margin-bottom: .8rem; text-transform: uppercase; }
.intro { color: #58685f; max-width: 600px; font-size: 1.05rem; line-height: 1.65;
    margin-bottom: 1.6rem; }
.method-strip { display: flex; gap: 1.5rem; flex-wrap: wrap; color: #456454;
    font-size: .8rem; padding: .8rem 0 1.8rem; }
.method-strip span { border-left: 2px solid #bdcebc; padding-left: .65rem; }
.step-label { color: #65776c; text-transform: uppercase; margin-bottom: .5rem; }
[data-testid="stFileUploaderDropzone"] {
    background: #f5f7f2; border: 1px dashed #a5baa9; border-radius: 8px;
    padding: 1.6rem 1rem; margin-top: .5rem;
}
.stButton button, [data-testid="stFileUploaderDropzone"] button {
    border-radius: 6px; transition: background-color .2s, border-color .2s;
}
[data-testid="stFileUploaderDropzone"] button {
    background: #285747; color: #fff; border-color: #285747;
}
[data-testid="stFileUploaderDropzone"] button:hover { background: #1e4437; color: #fff; }
[data-testid="stFileUploaderDropzone"] button p { color: inherit; }
button:focus-visible, a:focus-visible { outline: 3px solid #718d5c !important; outline-offset: 3px; }
.empty-result { min-height: 235px; display: flex; flex-direction: column;
    align-items: center; justify-content: center; text-align: center; padding: 1.2rem; }
.scan-mark { width: 66px; height: 80px; border: 1px solid #bac9bd;
    border-radius: 8px; padding: 12px; margin-bottom: 1.2rem; background: #f0f4ed; }
.empty-result strong { color: #344f41; font-weight: 500; }
.empty-result p { max-width: 240px; color: #65736b; font-size: .9rem; margin-top: .5rem; }
[data-testid="stMetricValue"] { color: #285747; font-variant-numeric: tabular-nums; }
[data-testid="stImage"] img { border-radius: 8px; max-height: 360px; object-fit: contain; }
.research-note { border-top: 1px solid #dce2d9; margin-top: 1.6rem;
    padding-top: 1rem; color: #5e6c63; font-size: .8rem; line-height: 1.65; }
.page-footer { display: flex; justify-content: space-between; gap: 1rem;
    flex-wrap: wrap; color: #66756a; font-size: .75rem; margin-top: 2rem; }
@media (max-width: 640px) {
    .block-container { padding: 2rem 1rem; }
    .masthead { margin-bottom: 2rem; align-items: flex-start; }
    .meta { max-width: 115px; text-align: right; }
    .method-strip { gap: .8rem; }
}
</style>
<div class="masthead">
  <div class="brand"><span class="brand-mark" aria-hidden="true">+</span> Federated Medical</div>
  <span class="meta">CHEST X-RAY WORKSPACE</span>
</div>
<div class="eyebrow">Medical imaging / Research demo</div>
""", unsafe_allow_html=True)
st.title("A clearer view of your chest X-ray.")
st.markdown("""
<p class="intro">Upload a chest X-ray to explore how a federated model distinguishes
normal images from images showing pneumonia.</p>
<div class="method-strip"><span>3 hospital datasets</span><span>2 image classes</span>
<span>One shared model</span></div>
""", unsafe_allow_html=True)


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

with st.spinner("Preparing the image analysis workspace..."):
    model = load_model()


def preprocess_image(img):
    # Match the notebook's bilinear resize on raw pixels, then its 1/99 scaling.
    pixels = tf.image.resize(np.asarray(img.convert("RGB")), (100, 100))
    return tf.expand_dims(pixels / 99.0, axis=0)


upload_column, result_column = st.columns([1.15, 1], gap="large")

with upload_column, st.container(border=True):
    st.markdown('<div class="step-label">01 / Input</div>', unsafe_allow_html=True)
    st.header("Your chest X-ray")
    st.caption("Choose a clear chest X-ray in PNG or JPEG format.")
    uploaded_img = st.file_uploader("Upload a Chest X-ray Image", type=["png", "jpg", "jpeg"])

with result_column, st.container(border=True):
    st.markdown('<div class="step-label">02 / Analysis</div>', unsafe_allow_html=True)
    st.header("Model prediction")
    st.caption("The predicted class and its model score appear here.")
    result_area = st.empty()
    result_area.markdown("""
    <div class="empty-result">
      <div class="scan-mark" aria-hidden="true">
        <svg viewBox="0 0 40 56" fill="none" stroke="#6b8a70" stroke-width="1.5">
          <path d="M20 3v31M16 12C8 12 3 25 4 40c0 7 10 10 13 3V17M24 12c8 0 13 13 12 28 0 7-10 10-13 3V17"/>
          <path d="M8 25h7M7 32h8M7 39h8M25 25h7M25 32h8M25 39h8"/>
        </svg>
      </div>
      <strong>Ready when you are</strong>
      <p>Upload an image to view the model's prediction.</p>
    </div>
    """, unsafe_allow_html=True)

if uploaded_img:
    img = Image.open(uploaded_img).convert("RGB")
    img_resized = img.resize((100,100))

    with upload_column:
        st.image(img, caption="Uploaded chest X-ray", use_container_width=True)

    x = preprocess_image(img)

    with result_column, st.spinner("Analyzing your X-ray..."):
        pred = model.predict(x)
    class_idx = np.argmax(pred)
    confidence = pred[0][class_idx]

    label = "Pneumonia" if class_idx == 1 else "Normal"

    with result_area.container():
        st.subheader(label)
        st.metric("Model score", f"{confidence*100:.2f}%")
        st.caption("This score reflects the model's output, not a clinical diagnosis.")

st.markdown("""
<div class="research-note"><strong>For research and demonstration.</strong>
This model achieved 82.5% accuracy on 624 project test images and can make mistakes.
Results require review by a qualified healthcare professional.</div>
<div class="page-footer"><span>Federated learning · Chest X-ray classification</span>
<span>Normal / Pneumonia</span></div>
""", unsafe_allow_html=True)
