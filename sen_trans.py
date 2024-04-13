import torch
from transformers import AutoTokenizer, AutoModel

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state  # Last layer hidden-states
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Sentences we want sentence embeddings for
sentences = ['This is an example sentence', 'Each sentence is converted']

# Models to use
models_to_use = [
    ("microsoft/codebert-base", "codebert_embeddings.txt"),
    ("roberta-base", "roberta_embeddings.txt"),
    ("sentence-transformers/xlm-r-distilroberta-base-paraphrase-v1", "xlmr_embeddings.txt")
]

# Iterate over models
for model_name, file_name in models_to_use:
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform mean pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Save embeddings to file
    with open(file_name, "w") as f:
        for embedding in sentence_embeddings:
            f.write(" ".join(map(str, embedding.tolist())) + "\n")

    print(f"Embeddings saved to {file_name}")

