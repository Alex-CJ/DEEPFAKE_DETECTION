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
        "AI-generated": 1 - prob_real,
    }


HERO_HTML = """
<div class="hero">
  <h1 class="hero-title">Deepfake Face Detector</h1>
  <p class="hero-sub">
    Upload a face photo and the model tells you whether it's a real
    photograph or an AI-generated one. It's the discriminator from a
    DCGAN-style network, trained adversarially from scratch and then
    fine-tuned on the
    <a href="https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces" target="_blank" rel="noopener">140k Real and Fake Faces</a>
    dataset &mdash; 92.9% accuracy and 0.982 AUC-ROC on a 20,000-image
    held-out test set.
  </p>
</div>
"""

LIMITS_HTML = """
<p class="hero-note">
  Trained on 64&times;64 face crops with no exposure to real-world
  artifacts like compression or motion blur, so accuracy on
  higher-resolution photos or newer generators (diffusion models,
  for instance) is untested.
</p>
"""

REAL_EXAMPLES = ["examples/real_1.jpg", "examples/real_2.jpg", "examples/real_3.jpg"]
FAKE_EXAMPLES = ["examples/fake_1.jpg", "examples/fake_2.jpg", "examples/fake_3.jpg"]
MORE_REAL_EXAMPLES = [f"examples/real_{i}.jpg" for i in range(4, 10)]
MORE_FAKE_EXAMPLES = [f"examples/fake_{i}.jpg" for i in range(4, 10)]

def _both(**tokens):
    """Force the same value in light and dark mode - this page has one
    deliberate identity and shouldn't flip to Gradio's default dark theme
    on visitors with a dark OS preference."""
    out = {}
    for key, value in tokens.items():
        out[key] = value
        out[f"{key}_dark"] = value
    return out


THEME = gr.themes.Base(
    font=[gr.themes.GoogleFont("IBM Plex Sans"), "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "monospace"],
).set(
    **_both(
        body_background_fill="#EEF0EA",
        body_text_color="#14181A",
        body_text_color_subdued="#4B5049",
        background_fill_primary="#EEF0EA",
        background_fill_secondary="#E4E7DD",
        block_background_fill="#F6F7F2",
        block_border_color="#C3C9B8",
        block_label_background_fill="#F6F7F2",
        block_label_text_color="#4B5049",
        block_title_text_color="#14181A",
        panel_background_fill="#E4E7DD",
        panel_border_color="#C3C9B8",
        color_accent_soft="#DEE6DB",
        border_color_accent="#1F6F52",
        border_color_accent_subdued="#9FB09F",
        button_primary_background_fill="#1F6F52",
        button_primary_background_fill_hover="#195A43",
        button_primary_text_color="#F6F7F2",
        button_primary_border_color="#1F6F52",
        button_secondary_background_fill="#F6F7F2",
        button_secondary_border_color="#C3C9B8",
        button_secondary_text_color="#14181A",
        input_background_fill="#FFFFFF",
        input_border_color="#C3C9B8",
    ),
    color_accent="#1F6F52",
    block_border_width="1px",
    block_label_text_weight="500",
    block_radius="4px",
    block_shadow="none",
    button_large_radius="4px",
    input_radius="4px",
    layout_gap="20px",
    block_padding="18px",
)

CSS = """
.gradio-container {
    max-width: 860px !important;
    margin-left: auto !important;
    margin-right: auto !important;
}

.hero { margin-bottom: 0.5rem; }
.hero-title {
    font-size: 2.25rem;
    font-weight: 600;
    line-height: 1.2;
    letter-spacing: -0.01em;
    color: #14181A;
    margin: 0 0 0.6rem 0;
}
.hero-sub {
    font-size: 1.05rem;
    line-height: 1.6;
    color: #3B4038;
    max-width: 66ch;
    margin: 0;
}
.hero-sub a { color: #1F6F52; }

.hero-note {
    font-size: 0.92rem;
    line-height: 1.55;
    color: #5B6156;
    max-width: 66ch;
    margin: 0.25rem 0 0 0;
    padding-top: 0.75rem;
    border-top: 1px solid #C3C9B8;
}
"""

with gr.Blocks(title="Deepfake Face Detector") as demo:
    gr.HTML(HERO_HTML)

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload a face photo")
            check_button = gr.Button("Check photo", variant="primary")
        with gr.Column(scale=1):
            result_output = gr.Label(num_top_classes=2, label="Result")

    check_button.click(fn=predict, inputs=image_input, outputs=result_output)

    with gr.Row():
        with gr.Column():
            gr.Examples(examples=REAL_EXAMPLES, inputs=image_input, examples_per_page=3,
                        label="Real photographs")
            with gr.Accordion("More real photographs", open=False):
                gr.Examples(examples=MORE_REAL_EXAMPLES, inputs=image_input,
                            examples_per_page=6)
        with gr.Column():
            gr.Examples(examples=FAKE_EXAMPLES, inputs=image_input, examples_per_page=3,
                        label="AI-generated photographs")
            with gr.Accordion("More AI-generated photographs", open=False):
                gr.Examples(examples=MORE_FAKE_EXAMPLES, inputs=image_input,
                            examples_per_page=6)

    gr.HTML(LIMITS_HTML)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, theme=THEME, css=CSS)
