# Data-Centric PAR with Synthetic Augmentation

This repository demonstrates our **three-step workflow** for improving Pedestrian Attribute Recognition (PAR) by selectively generating synthetic samples for underrepresented attributes, as described in our paper:

> **Rethinking Pedestrian Attribute Recognition with Synthetic Data Augmentation**  
> *(Add link or citation to out paper here.)*

## Table of Contents

1. [Step 1: Generate Synthetic Data via ComfyUI](#step-1-generate-synthetic-data-via-comfyui)  
2. [Step 2: Manual Checking & Labeling](#step-2-manual-checking--labeling)  
3. [Step 3: Train with Rethinking-of-PAR](#step-3-train-with-rethinking-of-par)  
4. [Notes & References](#notes--references)

---

## Step 1: Generate Synthetic Data via ComfyUI

1. **Identify “Weak” Attributes**  
   - Before generating synthetic data, decide which attributes need augmentation. Typically, these are attributes that are underrepresented or yield low performance in your baseline model.  
   - See **Sec. 3.1** of the paper for our criteria on selecting attributes.

2. **Clone ComfyUI & Install Requirements**  
   - We rely on [**ComfyUI**](https://github.com/comfyanonymous/ComfyUI) for text-to-image diffusion.  
   - After installing ComfyUI, also install additional dependencies from `requirements_generation.txt` in this repo.

3. **Load the Workflow & Wildcards**  
   - Import our `generation_workflow.json` in ComfyUI; it contains the pipeline for prompt generation, diffusion, and post-processing.  
   - If you use our wildcard setup, place them where ComfyUI can find them. However, **please note** that the files under `text_files/` are **not** for ComfyUI—they are used later by our labeling script.

4. **Generate Images**  
   - Configure the number of samples, noise levels, etc., in ComfyUI according to your needs.  
   - For guidance on how many synthetic images to create (e.g., 3× or 5× per real sample), see **Tables 6–7** in the paper.

---

## Step 2: Manual Checking & Labeling

1. **Clean the Images**  
   - Manually verify each generated batch to confirm the intended attribute is indeed present. Remove any incorrect or low-quality images.

2. **Add Labels & Create PKL**  
   - Run `add_synthetic_labels.py`, which uses the `.txt` files in `text_files/` to map each image’s prompt to an extended label scheme (`-1, 0, 1, 2, 3`).  
   - For details on what these label values represent, please see **Sec. 3.3** of the paper.  
   - This script merges your real + synthetic data into a single PKL for training.

3. **Example Data Links**  
   - We have sample generated images for five attributes uploaded here (fake placeholders):
     - Bald Head → [placeholder_LINK_BaldHead](LINK_BaldHead)  
     - Short Skirt → [placeholder_LINK_ShortSkirt](LINK_ShortSkirt)  
     - Age < 16 → [placeholder_LINK_Age16](LINK_Age16)  
     - Suit Up → [placeholder_LINK_SuitUp](_LINK_SuitUp)  
     - Plastic Bag → [placeholder_LINK_PlasticBag](LINK_PlasticBag)

---

## Step 3: Train with Rethinking-of-PAR

1. **Obtain the Baseline**  
   - Clone [**Rethinking-of-PAR**](LINK_TO_RETHINKING_PAR) as our baseline or integrate our code into your existing Rethinking-of-PAR setup.
   ```bash
   git clone LINK_TO_THIS_REPO_FOR_RETHINKING_PAR
   cd Rethinking-of-PAR
2. **Use Our PKL & Modified Loss**  
   - Update any config files to reference the PKL generated in Step 2.  
   - Make sure to use our modified BCE loss (e.g., `bceloss_augmented.py`) so it can handle the new label states (`-1` or `3`).

3. **Run Training**  
   ```bash
   python train_par.py --config config_par.yaml
   ```
   -  Adjust hyperparameters (batch size, learning rate, etc.) as desired. Refer to Sec. 4 of the paper for recommended settings and ablation results.
4. Evaluate
	-	Measure your final performance on the chosen PAR dataset (e.g., RAPv1, RAPv2, RAPzs). Compare it to a baseline to confirm the benefit of your synthetic augmentation.

---
## Notes & References

- **Recommended Augmentation Ratios**: Generating 3–5× synthetic images per real sample (i.e., 200–500% more data) often works well. See **Sec. 4.3** of our paper.  

---


