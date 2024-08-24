# Project Overview

**Why do we fine-tune a Language Model (LLM)?** Fine-tuning is essential when we need **to teach an LLM a new skill** or **enhance its understanding in a specific domain**. 

In this project, the goal was to fine-tune the SmolLM-135M-Instruct model to equip it with more reliable and accurate medical knowledge. Medicine is a vast and complex field that requires a deep understanding of numerous subjects, from basic medical sciences to advanced clinical skills.

To achieve this, I utilized the medalpaca/medical_meadow_medical_flashcards dataset, which is based on the Anki Medical Curriculum flashcards. These flashcards, created and continuously updated by medical students, cover a comprehensive range of topics, including anatomy, physiology, pathology, pharmacology, and more. They are designed to aid in the effective learning and retention of critical medical concepts through concise summaries and mnemonics.

### Result Demo:

**Question:** What is the relationship between very low Mg2+ levels, PTH levels, and Ca2+ levels?

**Correct Answer:** Very low Mg2+ levels correspond to low PTH levels which in turn results in low Ca2+ levels.

**Pre-trained Model's Answer:** The relationship between very low Mg2+ levels, PTH levels, and Ca2+ levels is complex and influenced by various factors. Here's a breakdown of the key factors: **Mg2+ levels:** 1. Low Mg

**Fine-tuned Model's Answer:** Very low Mg2+ levels, PTH levels, and Ca2+ levels are all related to the parathyroid gland. The parathyroid gland is a small gland located in the neck that produces parathyroid hormone (PTH) and calcium. When the parathyroid gland

**NOTE:** This result was obtained after only 5 epochs of training. Training for more epochs could lead to better results, of course.

## Setup
To run this project smoothly, Python 3.10 is required. The project has been tested and verified on Python 3.10.13. Follow the steps below to set up your environment:

1. Create a virtual environment:
    ```
    python -m venv env
    ```

2. Activate the virtual environment:
    ```
    source env/bin/activate
    ```

3. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```

4. Set up environment variables: Copy the .env.example file and rename it to .env:
    ```
    mv .env.example .env
    ```

## Run
You can explore the training steps using Jupyter notebooks:

1. Navigate to the notebooks directory:
    ```
    cd notebooks
    ```
2. Launch JupyterLab:
    ```chatinput
    jupyter-lab
    ```

Alternatively, you can start the fine-tuning process directly. The parameters used for fine-tuning are specified in the .env file, and explanations for these parameters can be found in comments within the same file. 
To begin the fine-tuning process, run:
```
python train.py
```