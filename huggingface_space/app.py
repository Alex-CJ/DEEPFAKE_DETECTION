"""
GAN-Discriminator Deepfake Detector - Gradio demo

Runs inference through ONNX Runtime instead of PyTorch. The full PyTorch
stack uses ~500MB of RAM just on import, which doesn't fit in a 512MB
hosting tier (Render's free plan); onnxruntime + numpy + pillow together
use a fraction of that. The .onnx file was exported once, offline, from
models/best_detector.pth (see export_onnx.py) - training still uses the
full PyTorch pipeline in detector.py, only this demo is PyTorch-free.
"""
import numpy as np
import onnxruntime as ort
import gradio as gr
from PIL import Image

MODEL_PATH = "best_detector.onnx"
IMAGE_SIZE = 64

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name


def preprocess(image: Image.Image) -> np.ndarray:
    """Match the val/test transform in detector.py: resize, [0,1], normalize to [-1,1]."""
    image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - 0.5) / 0.5
    array = array.transpose(2, 0, 1)  # HWC -> CHW
    return array[np.newaxis, ...]  # add batch dim


def predict(image):
    if image is None:
        return None

    input_tensor = preprocess(image)
    prob_real = float(session.run(None, {input_name: input_tensor})[0].item())

    return {
        "Real": prob_real,
        "AI-Generated / Fake": 1 - prob_real,
    }


DESCRIPTION = """
Upload a face photo and the model estimates whether it is a **real** photo
or an **AI-generated / GAN-produced** face.

This is the discriminator from a DCGAN-style network (with a self-attention
layer), trained from scratch adversarially and then fine-tuned on the
[140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
dataset. On its held-out test set it reached **92.9% accuracy** and an
**AUC-ROC of 0.982**.

**Limitations:** the model was trained on 64x64 face crops from GAN-generated
(mostly StyleGAN-style) images. Accuracy may be lower on non-face images, on
much higher resolution inputs, or on images produced by newer generators
(e.g. diffusion models) that it never saw during training.
"""

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a face image"),
    outputs=gr.Label(num_top_classes=2, label="Prediction"),
    title="Deepfake Face Detector",
    description=DESCRIPTION,
)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
