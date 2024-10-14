import torch
from transformers import BertTokenizer, BertForTokenClassification

entity_types = ["O", "B-MOUNTAIN", "I-MOUNTAIN"]

# Load the fine-tuned model and tokenizer
model = BertForTokenClassification.from_pretrained('fine_tuned_ner_model')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Function to perform inference
def infer_entities(text):
    # Tokenize the input text
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=tokenizer.model_max_length)
    
    # Ensure the model is in evaluation mode
    model.eval()

    with torch.no_grad():
        # Get the predictions
        outputs = model(**tokens)
        logits = outputs.logits

    # Get the predicted labels
    predicted_ids = torch.argmax(logits, dim=2)

    # Convert predicted IDs to labels
    predicted_labels = [entity_types[id.item()] for id in predicted_ids[0]]

    # Post-process to get the tokens with their corresponding labels
    tokenized_text = tokenizer.tokenize(text)
    result = []

    for token, label in zip(tokenized_text, predicted_labels):
        result.append((token, label))

    return result

# Example usage
text_to_infer =  "Mount Everest is the highest mountain in the world, located in the Himalayas Kilimanjaro is known for its snow-capped peak despite being near the equator The Andes stretch along the western coast of South America, creating a dramatic landscape Mount Fuji is a symbol of Japan and one of the most photographed mountains"
predicted_entities = infer_entities(text_to_infer)

# Display the results
for token, label in predicted_entities:
    if label != "O":
        print(f"{token}: {label}")