---
pipeline_tag: text-generation
tags:
  - gguf
  - llama-cpp
---

# NIL-3B GGUF

This repository contains the `nil-3b-q4_k_m.gguf` file for running the NIL-3B
project locally with a GGUF-compatible inference runtime.

## Local inference

Install the project's dependencies, then run:

```bash
python -m pip install -r requirements.txt
python -m src.chat
```

The chat script expects `nil-3b-q4_k_m.gguf` in the project root.

## Uploading this model

If you do not already have a `.env`, copy `.env.example` to `.env`. Set
`HF_TOKEN` to a Hugging Face token with write access and `HF_REPO_ID` to the
intended `owner/repository`. The `.env` file is ignored by Git and must not be
committed or shared.

Install the uploader dependencies if they are not already installed:

```bash
python -m pip install huggingface_hub python-dotenv
```

Run the uploader from any working directory:

```bash
python upload_hf.py
```

It creates the model repository if needed and uploads the GGUF file. Review the
target repository and its visibility on Hugging Face before running the upload.

## Licensing and model details

The root `LICENSE` covers the project source code; it does not by itself confirm
the license or redistribution rights for the model weights or training data.
Confirm those rights and document the model's base model, training data, and
supported capabilities before making this repository public.
