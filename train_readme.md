# DETR Fine-Tuning on Kaohsiung Port Dataset

This repository contains code to fine-tune the [DETR (DEtection TRansformer)](https://arxiv.org/abs/2005.12872) model on a custom object detection dataset from the Kaohsiung Port.

---

## 📦 Features

- Fine-tunes `facebook/detr-resnet-50` using Hugging Face Transformers
- Supports dataset loading from Hugging Face Hub (`jsonl` or `parquet`)
- Automatically pushes models to Hugging Face Hub after every epoch
- Supports resume-from-checkpoint functionality
- Configurable image size, batch size, learning rate, and more

---


