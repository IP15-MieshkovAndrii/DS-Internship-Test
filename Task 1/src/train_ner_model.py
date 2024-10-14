from transformers import BertTokenizer, BertForTokenClassification, AdamW
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm 
import torch
import pandas as pd

entity_types = ["O", "B-MOUNTAIN", "I-MOUNTAIN"]

num_labels = len(entity_types)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 
model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

batch_size = 32

learning_rate = 5e-5

file_path = '../data/labeled_mountain_dataset.csv'
df = pd.read_csv(file_path)

train_dataset_sample = []
current_sentence = []
current_entities = []

for index, row in df.iterrows():
    word = row['Word']
    tag = row['Tag']
    
    if pd.isna(word) or pd.isna(tag):
        if current_sentence:
            train_dataset_sample.append({
                "text": " ".join(current_sentence),
                "labels": {"entities": current_entities}
            })
            current_sentence = []
            current_entities = []
    else:
        current_sentence.append(word)
        
        complete_sentence = " ".join(current_sentence)
        
        start_pos = len(complete_sentence) - len(word)
        end_pos = len(complete_sentence)
        
        if tag == "B-MOUNTAIN":
            current_entities.append((start_pos, end_pos, "MOUNTAIN"))
        elif tag == "I-MOUNTAIN" and current_entities:
            last_entity = current_entities[-1]
            current_entities[-1] = (last_entity[0], end_pos, last_entity[2])

if current_sentence:
    train_dataset_sample.append({
        "text": " ".join(current_sentence),
        "labels": {"entities": current_entities}
    })

print(train_dataset_sample)


def tokenize_and_format_data(dataset, tokenizer):
    tokenized_data = []
    for sample in dataset:
        text = sample["text"]
        entities = sample["labels"]["entities"]

        # Tokenize the input text using the BERT tokenizer
        tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text)))
        # Initialize labels for each token as 'O' (Outside)
        labels = ['O'] * len(tokens)

        # Update labels for entity spans
        for start, end, entity_type in entities:
            # Tokenize the prefix to get the correct offset
            prefix_tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text[:start])))
            start_token = len(prefix_tokens)

            # Tokenize the entity to get its length
            entity_tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text[start:end])))
            end_token = start_token + len(entity_tokens) - 1

            # Check if start_token and end_token are within bounds
            if start_token < len(labels):
                labels[start_token] = "B-MOUNTAIN"
            else:
                print(f"Warning: start_token {start_token} is out of range for text: '{text}'")
                continue

            # Ensure end_token does not exceed the length of labels
            for i in range(start_token + 1, min(end_token + 1, len(labels))):
                labels[i] = "I-MOUNTAIN"

        # Convert tokens and labels to input IDs and label IDs
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        label_ids = [entity_types.index(label) for label in labels]

        # Pad input_ids and label_ids to the maximum sequence length
        padding_length = tokenizer.model_max_length - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * padding_length
        label_ids += [entity_types.index('O')] * padding_length

        tokenized_data.append({
            'input_ids': input_ids,
            'labels': label_ids
        })

    # Convert tokenized data to PyTorch dataset
    dataset = TensorDataset(
        torch.tensor([item['input_ids'] for item in tokenized_data]),
        torch.tensor([item['labels'] for item in tokenized_data])
    )
    return dataset

train_data = tokenize_and_format_data(train_dataset_sample, tokenizer)
train_dataloader = DataLoader(train_data, batch_size=batch_size)

optimizer = AdamW(model.parameters(), lr=learning_rate)
num_epochs = 15 

for epoch in range(num_epochs):
    model.train()
    for batch in tqdm(train_dataloader, desc="Training"):
        inputs, labels = batch
        # Unpack the tuple
        outputs = model(inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

model.save_pretrained('fine_tuned_ner_model')