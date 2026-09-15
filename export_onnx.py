"""
One-off dev utility: export the trained discriminator to ONNX so the web
demo (huggingface_space/) can run inference without depending on PyTorch,
which is too heavy in RAM for a 512MB hosting tier. Run locally, once,
whenever models/best_detector.pth changes:

    python export_onnx.py
"""
import torch
from detector import Discriminator

CHECKPOINT_PATH = "models/best_detector.pth"
OUTPUT_PATH = "huggingface_space/best_detector.onnx"

device = torch.device("cpu")
model = Discriminator().to(device)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

dummy_input = torch.randn(1, 3, 64, 64)
torch.onnx.export(
    model,
    dummy_input,
    OUTPUT_PATH,
    input_names=["image"],
    output_names=["prob_real"],
    dynamic_axes={"image": {0: "batch"}, "prob_real": {0: "batch"}},
    opset_version=17,
    dynamo=False,
)

print(f"Exported {CHECKPOINT_PATH} -> {OUTPUT_PATH}")
