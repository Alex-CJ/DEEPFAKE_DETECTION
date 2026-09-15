---
title: Deepfake Face Detector
emoji: 🕵️
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
---

# Deepfake Face Detector

Upload a face image and this app tells you whether the discriminator
thinks it's a real photo or an AI-generated (GAN) face.

Model: a DCGAN-style discriminator with self-attention, trained
adversarially then fine-tuned on the
[140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
dataset. Test set: 92.9% accuracy, AUC-ROC 0.982.

Full project, training code, and report: see the main GitHub repository.
