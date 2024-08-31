# Project Overview

**Why do we fine-tune a Language Model?** Fine-tuning is essential when we need **to teach an LLM a new skill** or **enhance its understanding in a specific domain**. 

In this project, we aim to fine-tune both small and large instruct/chat language models, including SmolLM for small language models (SLM) and Mistral for large language models (LLM).

The primary example dataset used is `medalpaca/medical_meadow_medical_flashcards`, derived from the Anki Medical Curriculum flashcards. These flashcards, created and regularly updated by medical students, cover a broad spectrum of medical topics—such as anatomy, physiology, pathology, and pharmacology—and are designed to support effective learning and retention through concise summaries and mnemonics.

Although this project uses the medical_meadow_medical_flashcards dataset, it is adaptable to other datasets for question/answering and instruct/response tasks. Integration of different datasets can be easily accomplished by modifying the configuration files.

The project also supports quantization, making it feasible to fine-tune large models like Mistral and Llama on more modest hardware setups.

The fine-tuned SmolLM model is available on [HuggingFace 🤗](https://huggingface.co/mohres/SmolLM-135M-Instruct-medical_meadow_medical_flashcards-10epochs)

## Model Performance Comparison (SmolLM-135M-Instruct)

While the base model's responses are general and follow a markdown format, the fine-tuned model appears to have effectively incorporated the specialized skills it was trained on, particularly in replicating the expected response style. However, as someone without a medical background, I am unable to assess the accuracy or quality of the responses provided by either model. Below is a table comparing the responses of the original model, base model, and fine-tuned model for a set of medical questions.

| Question                        | Original Response                  | Base Model Response                                                                                                                                                                                                                                                                                                                                   | Fine-tuned Model Response                                                                                                       |
|---------------------------------|------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| What antimalarial medication has activity against Plasmodium hypnozoites? | Primaquine is the antimalarial medication that has activity against Plasmodium hypnozoites.    | Antimalarial medications are a class of drugs that are effective against Plasmodium parasites, which are responsible for causing malaria. The most common antimalarial medications used to treat malaria include: 1. **Ampicillin (Amphotericin B)**: A broad-spectrum antibiotic that is effective against Plasmodium species. 2. **Fosamivir (Fosamivir)**: A broad-spectrum antiviral that is effective against Plasmodium species. 3. **Influvir (Influvir)**: A broad-spectrum antiviral that is effective against Plasmodium species. 4. **Ampicillin | Erythromycin is the antimalarial medication that has activity against Plasmodium hypnozoites.                                   |
| What is the term used to describe the amount of substance filtered in Bowman's space per unit time? | The amount of substance filtered in Bowman's space per unit time is called the filtered load. | The term "Bowman's space" is a colloquialism used to describe the amount of substance filtered in a Bowman's space, which is a hypothetical space where a spacecraft is launched into orbit. The concept is based on the idea that a spacecraft can be launched into orbit and then return to Earth, with the spacecraft filtering out any remaining particles or debris that may have been present in the space. The term "Bowman's space" is often used to describe the amount of substance filtered in a Bowman's space, which is a hypothetical space where a spacecraft is launched into orbit and then returns to Earth. The                                | The term used to describe the amount of substance filtered in Bowman's space per unit time is called the filtration efficiency. |
| What is the term used to describe the blood volume contained in the veins?  | The blood volume contained in the veins is called the unstressed volume. | The term "blood volume" refers to the amount of blood that circulates through the body's circulatory system. It is a complex process that involves the exchange of oxygen, nutrients, and waste products between the blood and the tissues. The blood volume is typically measured in liters (L) or milliliters (mL) and is typically distributed throughout the body through the circulatory system. There are several key components that contribute to the blood volume: 1. **Red blood cells**: These are the cells that carry oxygen from the lungs to the tissues. They have a diameter of about 1.5 mm and are composed of hemoglobin                                                                                                                                                                                                                            | The term used to describe the blood volume contained in the veins is called venous blood.                                       |
| What is the most important factor for determining the prognosis of a transitional cell carcinoma? | The most important factor for determining the prognosis of a transitional cell carcinoma is the degree of invasion. | Determining the prognosis of a transitional cell carcinoma (TCC) is crucial, as it can significantly impact the patient's quality of life, treatment options, and overall survival. The most important factor for determining the prognosis of a TCC is the presence of a clear diagnosis, a well-defined tumor, and a favorable prognosis. Here are some key factors that can influence the prognosis of a TCC: 1. **Clear diagnosis**: A TCC is diagnosed with a clear tumor, which means it has a benign or benign-like growth. A TCC is also diagnosed with a favorable prognosis, as it is unlikely to spread to                                                                                   | The most important factor for determining the prognosis of a transitional cell carcinoma is the presence of a high grade.       |

### Observations

**Accuracy of Responses:** The fine-tuned model replicates the expected format and style but demonstrates some inaccuracies. For example, erythromycin is incorrectly identified as an antimalarial medication effective against Plasmodium hypnozoites. The correct answer should be primaquine.

**Base Model Issues:** The base model's responses are overly generalized and sometimes off-topic, especially in medical contexts, as seen in the incorrect explanations about Bowman's space and blood volume.

**NOTE:** This result was obtained after only 10 epochs of training. Training for more epochs may lead to better results ¯\\_(ツ)_/¯.

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
    ```
    jupyter-lab
    ```

Alternatively, you can start the fine-tuning process directly. The parameters used for fine-tuning are specified in the configuration files inside `configs` folder. 
To begin the fine-tuning process with the default SmalLM model, run:
```
python train.py
```

If you want to train a different model, you can specify the model configuration by passing the config file name as an argument:
```
python train.py --model Mistral
```

Replace Mistral with the name of the desired model's config file. To train a new model, simply create a new config file with the desired settings and run train.py with the appropriate model name.