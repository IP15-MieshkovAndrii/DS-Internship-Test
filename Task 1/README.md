# Named Entity Recognition (NER) for Mountain Names

This project fine-tunes a BERT-based model for Named Entity Recognition (NER) to identify mountain names in a given text. The model has been trained to recognize mountain names using a custom dataset of sentences that mention various famous mountains.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Setup Instructions](#setup-instructions)
3. [Data Preparation](#data-preparation)
4. [Training the Model](#training-the-model)
5. [Inference (Using the Model)](#inference-using-the-model)
6. [Potential Improvements](#potential-improvements)

## Project Overview

This project focuses on training a Named Entity Recognition (NER) model to detect mountain names within text. The model is built using the pre-trained BERT (Bidirectional Encoder Representations from Transformers) from the Hugging Face `transformers` library and fine-tuned on a dataset of sentences with mountain names.

The main components of this project are:
- Data preprocessing and labeling
- Model fine-tuning using BERT
- Model inference for identifying mountain names in unseen sentences

## Setup Instructions

### 1. Clone the Repository

First, clone this repository to your local machine:

```bash
git clone https://github.com/IP15-MieshkovAndrii/DS-Internship-Test
cd Task 1
```

### 2. Install Dependencies

Use `pip` to install the required libraries by running:

```bash
pip install -r requirements.txt
```

This will install the following packages:
- `torch` (for PyTorch-based deep learning)
- `transformers` (for BERT and other transformer models)
- `tqdm` (for progress bars)
- `pandas` (for data manipulation)

### 3. Download or Prepare the Dataset

The dataset is stored in the `data/` folder as `labeled_mountain_dataset.csv`. It contains sentences where mountain names are labeled using the IOB (Inside-Outside-Beginning) tagging scheme. You can view the dataset creation process in `data_creation.ipynb`.

### 4. Download Pre-trained Model

The BERT model pre-trained for NER has been fine-tuned on the custom mountain dataset. The weights for this model are saved in the `models/` directory. If you'd like to fine-tune the model yourself, refer to the next section on how to train the model.

## Data Preparation

The dataset consists of sentences where mountain names are tagged using the IOB format. For example:
```
Mount B-MOUNTAIN
Everest I-MOUNTAIN
is O
the O
highest O
mountain O
...
```

The data is loaded and processed in `train_ner_model.py` for fine-tuning the model. Tokenization is performed using BERT's tokenizer, and the tags are aligned to match tokenized words.

## Training the Model

To fine-tune the BERT model on the dataset, run the `train_ner_model.py` script:

```bash
python train_model.py
```

The script:
1. Loads and tokenizes the dataset.
2. Fine-tunes the BERT model for token classification using the mountain dataset.
3. Saves the fine-tuned model to the `models/fine_tuned_ner_model/` directory.

You can modify the hyperparameters (e.g., batch size, number of epochs) in `train_ner_model.py` as needed.

## Inference (Using the Model)

To use the fine-tuned model for predicting mountain names in new sentences, run the `inference.py` script:

```bash
python inference.py
```

The script:
1. Loads the fine-tuned model from the `fine_tuned_ner_model/` directory.
2. Runs inference on the input text and prints the identified mountain names.

You can modify the input sentences directly in the script or provide text as an argument.

## Potential Improvements

1. **Data Augmentation**: Increase the size of the dataset by generating more sentences with diverse mountain mentions to improve the model’s generalization ability.
2. **Entity-Level F1 Score**: Evaluate the model using the entity-level F1 score to measure how well it identifies mountain names.
3. **Hyperparameter Tuning**: Experiment with different learning rates, batch sizes, and number of epochs for improved model performance.
4. **Domain-Specific Pre-training**: Fine-tune BERT on a large corpus of geographical or hiking-related text before performing NER, to enhance model understanding of the domain.

---

This **README** provides a comprehensive overview of how to set up, run, and use the project. You can adjust the explanations or details based on your specific implementation and needs.
