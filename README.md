# GAN-Based Deepfake Face Detector

A university project (COMP1827) evaluating whether the discriminator from a
DCGAN-style network — trained adversarially and then fine-tuned — can be
repurposed as a deepfake/AI-generated face detector.

**Try it live:** _add your Render URL here once deployed, e.g.
`https://deepfake-face-detector.onrender.com`_

Full write-up: [`001366584_COMP1827_REPORT.pdf`](./001366584_COMP1827_REPORT.pdf)

## Results

Trained and fine-tuned on the
[140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
dataset (100,000 train / 20,000 validation / 20,000 test images, balanced
real/fake, no leakage between splits).

| Split      | Accuracy | Precision | Recall | F1     | AUC-ROC |
|------------|---------:|----------:|-------:|-------:|--------:|
| Test       | 92.92%   | 95.08%    | 90.51% | 92.74% | 0.9823  |
| Validation | 90.71%   | 89.20%    | 92.62% | 90.88% | 0.9697  |
| Train      | 92.17%   | 90.92%    | 93.70% | 92.29% | 0.9776  |

Full metrics: [`models/evaluation_metrics.json`](./models/evaluation_metrics.json),
plots: [`plots/`](./plots).

## How it works

1. **Phase 1 — adversarial pretraining.** A generator and discriminator
   (both with a self-attention block, spectral normalization on the
   discriminator) are trained from scratch as a standard GAN on 64x64 face
   images.
2. **Phase 2 — supervised fine-tuning.** The discriminator from the best
   Phase 1 checkpoint is fine-tuned directly as a binary real/fake
   classifier on the labeled dataset.
3. Only the discriminator is used at inference time — that's the "deepfake
   detector".

See the report for background, architecture details, and discussion of
limitations (low input resolution, no compression/lighting/motion-blur
robustness testing).

## Project structure

```
detector.py            Training/evaluation pipeline (see "Usage" below)
data/                   Full dataset: train/val/test, each with real/ and fake/
data_testing/           Small subset for a quick smoke-test run
models/                 Trained checkpoints + evaluation_metrics.json + training_history.json
saved_models/           Earlier GAN-only checkpoints (generator/discriminator)
plots/                  Training curves, metric plots, confusion matrix inputs
generated_images/       Sample generator outputs per epoch
huggingface_space/      Self-contained Gradio web demo (see below)
```

`stylegan2-ada-pytorch/` (vendored NVIDIA StyleGAN2-ADA repo, reference/
tooling only, not part of the trained pipeline) is excluded from this repo
via `.gitignore` since it's its own git repo. Clone it separately if you
need it: `git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git`

`ffhq.pkl` and `fast_discriminator.pth` at the project root are earlier
exploratory files (experimenting with a pretrained StyleGAN2-FFHQ
discriminator) and are **not** used by the final pipeline described in the
report or in `detector.py`.

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ (developed and tested with PyTorch 2.9, CPU-only).

## Usage

Train from scratch (both phases) and evaluate on all splits:

```bash
python detector.py --mode train --data_dir data --epochs 20 --batch_size 32
```

For a quick run on the small sample set instead of the full 140k images:

```bash
python detector.py --mode train --data_dir data_testing --epochs 5 --batch_size 64
```

Evaluate an existing checkpoint on the test set only (fast, no training):

```bash
python detector.py --mode evaluate --model_path models/best_detector.pth
```

## Web demo

`huggingface_space/` (named for its original target, now deployed via
Render instead — see below) contains a small Gradio app that loads
`models/best_detector.pth` and lets you upload an image to get a Real /
AI-Generated prediction with confidence scores.

Run it locally:

```bash
cd huggingface_space
pip install -r requirements.txt
python app.py
```

Then open the local URL it prints (default `http://127.0.0.1:7860`).

### Deploying to Render (free)

Hugging Face Spaces now requires a paid plan to run Gradio/Docker apps on
free personal accounts (static-only Spaces stay free, but they can't run
Python). Render's free web service tier still runs plain Python for free,
so that's what this project deploys to instead:

1. Push this repository to GitHub (see below).
2. Go to [dashboard.render.com](https://dashboard.render.com) → **New** →
   **Web Service**, and connect your GitHub repo.
3. Configure the service:
   - **Root Directory:** `huggingface_space`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python app.py`
   - **Instance Type:** Free
4. Deploy. Render gives you a public URL like
   `https://<service-name>.onrender.com` — paste it into the "Try it live"
   link at the top of this README.

Note: on the free tier, the service spins down after 15 minutes of no
traffic, so the first request after a while takes ~30-60 seconds to wake
up — expected behavior, not a bug.

**Note on committing to GitHub:** several checkpoint files in this repo
(`models/*.pth`, `huggingface_space/best_detector.pth`, `ffhq.pkl`) are tens
to hundreds of MB. GitHub will still accept the ~49MB checkpoints but warns
above 50MB and hard-blocks above 100MB — `ffhq.pkl` (~380MB) will need
[Git LFS](https://git-lfs.com) or to be excluded via `.gitignore` if you
push this repo as-is.

## Acknowledgements

Group project — see the report for full author list and contributions. AI
tools were used to assist with wording, formatting, and LaTeX for the
written report; experimental design, coding, and analysis are the authors'
own.
