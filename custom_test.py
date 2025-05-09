import sys
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

model_name = sys.argv[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)
model.eval()  
dataset = load_dataset("dogtooth/default_project_dev_test", split="dev_test")

total_loss = 0.0
total_tokens = 0
chunk_size = 1024
stride = 256

for example in dataset:
    text = example["text"]
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    # Create sliding window chunks
    chunks = []
    for i in range(0, len(tokens), stride):
        chunk = tokens[i:i+chunk_size]
        if len(chunk) == chunk_size:
            chunks.append(chunk)
    
    # Process chunks in batches matching training
    batch_size = 8  # Match training batch size
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        
        # Create batch with padding
        inputs = torch.full((len(batch_chunks), chunk_size), tokenizer.eos_token_id)
        for j, chunk in enumerate(batch_chunks):
            inputs[j] = torch.tensor(chunk)
        
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs, labels=inputs)
        
        # Calculate loss only on valid positions
        valid_positions = (inputs != tokenizer.eos_token_id)
        loss = outputs.loss * valid_positions.sum()
        
        total_loss += loss.item()
        total_tokens += valid_positions.sum().item()

avg_loss = total_loss / total_tokens
perplexity = math.exp(avg_loss)
print("Perplexity:", perplexity)