"""
GAN-Discriminator Deepfake Detector - Gradio demo

Loads the trained discriminator (models/best_detector.pth from the main
project, copied alongside this file as best_detector.pth) and serves it
through a simple image-upload interface.

Model architecture is duplicated from detector.py so this folder can be
deployed on its own (e.g. as a Hugging Face Space) without depending on
the rest of the repository.
"""
import os
import torch
import torch.nn as nn
import gradio as gr
from torchvision import transforms

MODEL_PATH = "best_detector.pth"
IMAGE_SIZE = 64


# ============================================================================
# NETWORK ARCHITECTURE (must match detector.py exactly to load the weights)
# ============================================================================
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, channels, height, width = x.size()
        query = self.query(x).view(batch, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch, -1, height * width)
        attention = torch.softmax(torch.bmm(query, key), dim=-1)
        value = self.value(x).view(batch, -1, height * width)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch, channels, height, width)
        return self.gamma * out + x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        norm = nn.utils.spectral_norm

        self.features = nn.Sequential(
            norm(nn.Conv2d(3, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2),

            norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            norm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            SelfAttention(256),

            norm(nn.Conv2d(256, 512, 4, 2, 1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.3),

            norm(nn.Conv2d(512, 1024, 4, 2, 1)),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.3)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 2 * 2, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.features(x)
        return self.classifier(features)


# ============================================================================
# MODEL LOADING
# ============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Discriminator().to(device)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Same normalization used for val/test data in detector.py
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


# ============================================================================
# PREDICTION
# ============================================================================
def predict(image):
    if image is None:
        return None

    img_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        prob_real = model(img_tensor).item()

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
    # Render (and most PaaS hosts) assign the port via $PORT and expect the
    # server to listen on 0.0.0.0, not just localhost.
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
