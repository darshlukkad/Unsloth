# Unsloth — GRPO & LoRA Colab Notebooks

This repository contains Colab notebooks and example code for training and evaluating small language models (SmolLM2) with LoRA adapters and GRPO (Generative Reward Policy Optimization). The notebooks were developed as part of an assignment/demo and include setup, training, adapter saving, merging, and export steps.

Drive with explanation videos
--------------------------------

All explanation and walkthrough videos for the notebooks are available in this Google Drive folder:

https://drive.google.com/drive/folders/1nFFbjTyXh3QrfVGRulyhMrFCjX-x3hxN?usp=sharing

Notebooks included
------------------

- `colab_2_lora_smollm2.ipynb` — LoRA fine-tuning examples for SmolLM2.
- `colab3_dpo_pref.ipynb` — DPO / preference-learning experiments.
- `colab4_grpo.ipynb` — Full GRPO training demo (includes building rewards, training LoRA adapters, merging and exporting to GGUF).
- `colab_5.ipynb` and `Colab_Full_finetune(SmolLM2_135M).ipynb` — additional demos and full-finetuning examples.

Quick overview of `colab4_grpo.ipynb`
----------------------------------

- Installs required packages (Unsloth, TRL, Transformers, PEFT, bitsandbytes, etc.).
- Loads a SmolLM2 base model in 4-bit for low-VRAM use via `Unsloth.FastLanguageModel`.
- Attaches LoRA adapters and runs a GRPO training loop on a small slice of GSM8K with custom reward functions (correctness + format).
- Saves only the LoRA adapter (compact), then demonstrates how to merge adapters into a single model and export to GGUF for Ollama.

How to run (Colab)
-------------------

1. Open the desired notebook in Colab (the top of each notebook includes a Colab badge/button).
2. Run the cells in order. The notebooks set stability environment flags before importing `transformers`/`trl`/`peft` to avoid runtime issues on Colab.
3. For `colab4_grpo.ipynb` specifically, run the cells that install packages and set environment flags first, then the model/load and GRPO training cells.

How to run locally (recommended: Linux/macOS with CUDA GPU)
-----------------------------------------------------------

1. Create a Python environment (3.10+ recommended):

	python -m venv .venv
	source .venv/bin/activate

2. Install the core dependencies used by the notebooks:

	pip install -U unsloth trl transformers datasets accelerate peft bitsandbytes einops evaluate sentencepiece

3. Open the notebook in Jupyter or run the steps in a Python script. Ensure you have a GPU with sufficient memory for 4-bit + LoRA. For low-VRAM setups the notebooks use 4-bit quantization and small batch sizes.

Notes and tips
--------------

- The notebooks use `Unsloth` to patch Transformers for low-VRAM workflows. Import `unsloth` before `transformers`/`trl`/`peft` as shown in the notebooks.
- If you only want to use or share the trained policy, the notebooks save only the LoRA adapter directory (e.g., `smollm2_grpo_adapter`). This is small and can be later merged into the base weights and exported (GGUF) for inference.
- GPU recommended. If you run on CPU, training will be very slow; adjust batch sizes and sequence lengths accordingly.

Contact / Attribution
---------------------

This repository and notebooks were developed as an educational/demo assignment. For questions about the notebooks or issues running them, check the explanation videos in the Drive link above or open an issue in this repository.

Enjoy the demos!
