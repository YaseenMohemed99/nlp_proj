import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_scheduler
from transformers.modeling_utils import Conv1D
from datasets import load_dataset
import argparse
from tqdm import tqdm
from functools import partial
import time
from datetime import timedelta
import re
import contractions  
import pkg_resources
from symspellpy import SymSpell
from multiprocessing import Pool, cpu_count
import math
import torch.nn.functional as F

class TextNormalizer:
    _instance = None
    
    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.sym_spell = SymSpell(max_dictionary_edit_distance=2)
            dictionary_path = "frequency_dictionary_en_82_765.txt"
            cls._instance.sym_spell.load_dictionary(dictionary_path, 0, 1)
        return cls._instance

    def correct_typos(self, text):
        suggestions = self.sym_spell.lookup_compound(text, max_edit_distance=2)
        return suggestions[0].term if suggestions else text


    def normalize(self,text):
        text = self.fix_contractions(text)
        text = self.correct_typos(text)
        text = self.clean_whitespace(text)
        return text

    @staticmethod
    def fix_contractions(text):
        return contractions.fix(text)


    @staticmethod
    def clean_whitespace(text):
        return " ".join(text.split())

def deduplicate_texts(texts):
    """Remove duplicate texts while preserving order"""
    seen = set()
    return [text for text in texts if not (text in seen or seen.add(text))]
def init_pool():
    global normalizer
    normalizer = TextNormalizer()

def normalize_text(text):
    return normalizer.normalize(text)

class CustomGPT2(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.dropout_type = "none"
        self.dropout_rate = 0.0

    def set_dropout_strategy(self, strategy, rate):
        self.dropout_type = strategy
        self.dropout_rate = rate
        
        if strategy == "scaled_last":
            self._scale_last_layer(rate)

    def _scale_last_layer(self, rate):
        """Properly handle GPT-2's layer dimensions"""
        d_model = self.config.n_embd  
        
        scaled_dim = int((4 * d_model) / (1 - rate))
        
        last_layer = self.transformer.h[-1].mlp
        device = last_layer.c_fc.weight.device

        in_features = last_layer.c_fc.weight.shape[0]  

        last_layer.c_fc = Conv1D(scaled_dim, in_features).to(device)
        last_layer.c_proj = Conv1D(d_model, scaled_dim).to(device)
        
        torch.nn.init.normal_(last_layer.c_fc.weight, std=0.02)
        torch.nn.init.normal_(last_layer.c_proj.weight, 
                            std=0.02/math.sqrt(2 * self.config.n_layer))
        
        if last_layer.c_fc.bias is not None:
            torch.nn.init.zeros_(last_layer.c_fc.bias)
        if last_layer.c_proj.bias is not None:
            torch.nn.init.zeros_(last_layer.c_proj.bias)

    def modify_forward(self):
        """Modified to handle all transformer arguments"""
        original_forward = self.transformer.forward
        
        def custom_forward(input_ids=None, attention_mask=None, **kwargs):
            assert input_ids.device == self.device, \
                f"Input on {input_ids.device} vs model on {self.device}"
                
            outputs = original_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )
            
            if self.dropout_type == "all":
                outputs.last_hidden_state = F.dropout(
                    outputs.last_hidden_state,
                    p=self.dropout_rate,
                    training=self.training
                )
            elif self.dropout_type == "ffn":
                # Add dropout to each FFN output
                for layer in self.transformer.h:
                    layer.mlp = torch.nn.Sequential(
                        layer.mlp,
                        torch.nn.Dropout(p=self.dropout_rate)
                    )
            elif self.dropout_type == "last_layer":
                # Apply dropout to entire last layer output
                outputs.last_hidden_state = F.dropout(
                outputs.last_hidden_state,  
                p=self.dropout_rate,
                training=self.training
                )
            return outputs
        
        self.transformer.forward = custom_forward


class ChunkedDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, chunk_size=1024, stride=512, normalization="raw"):
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.stride = stride
        self.chunks = []
                
        # Apply normalization
        if normalization != "raw":
            if normalization == "normalized":
                with Pool(processes=cpu_count(), initializer=init_pool) as pool:
                    processed_texts = list(tqdm(pool.imap(normalize_text, texts), 
                                            total=len(texts), 
                                            desc="Normalizing texts"))
            elif normalization == "deduplicated":
                with Pool(processes=cpu_count(), initializer=init_pool) as pool:
                    normalized_texts = list(tqdm(pool.imap(normalize_text, texts), 
                                            total=len(texts), 
                                            desc="Normalizing texts"))
                processed_texts = deduplicate_texts(normalized_texts)
        else:
            processed_texts = texts
        
        # Preprocess and chunk all texts
        for text in processed_texts:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            for i in range(0, len(tokens), stride):
                chunk = tokens[i:i+chunk_size]
                if len(chunk) == chunk_size:
                    self.chunks.append(chunk)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        return {
            'input_ids': torch.tensor(chunk),
            'labels': torch.tensor(chunk)
        }

def dynamic_collate_fn(batch, pad_token_id):
    max_length = max(len(item['input_ids']) for item in batch)
    padded_batch = {
        'input_ids': [],
        'attention_mask': [],
        'labels': []
    }
    
    for item in batch:
        pad_length = max_length - len(item['input_ids'])
        padded_batch['input_ids'].append(
            torch.cat([item['input_ids'], torch.full((pad_length,), pad_token_id)]))
        padded_batch['attention_mask'].append(
            torch.cat([torch.ones_like(item['input_ids']), torch.zeros(pad_length,)]))
        padded_batch['labels'].append(
            torch.cat([item['labels'], torch.full((pad_length,), -100)]))
    
    return {
        'input_ids': torch.stack(padded_batch['input_ids']),
        'attention_mask': torch.stack(padded_batch['attention_mask']),
        'labels': torch.stack(padded_batch['labels'])
    }

def train(model, train_loader, val_loader, args, tokenizer):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
    total_start_time = time.time()
    epoch_times = []
    # Calculate total training steps
    best_val_loss = float('inf')
    total_steps = len(train_loader) * args.epochs
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    num_warmup_steps = int(0.1 * total_steps)
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps
    )
    
    global_step = 0
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            if args.max_steps > 0 and global_step >= args.max_steps:
                break
            
            inputs = batch['input_ids'].to(args.device)
            masks = batch['attention_mask'].to(args.device)
            labels = batch['labels'].to(args.device)
            
            with torch.cuda.amp.autocast(enabled=args.fp16):
                outputs = model(
                    input_ids=inputs,
                    attention_mask=masks,
                    labels=labels,
                    use_cache=False  
                )
                loss = outputs.loss / args.gradient_accumulation_steps
                
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                lr_scheduler.step()
                
            total_loss += loss.item() * args.gradient_accumulation_steps
            global_step += 1
            
            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        # Validation and logging 
        epoch_duration = time.time() - epoch_start_time
        epoch_times.append(epoch_duration)
        avg_loss = total_loss / len(train_loader)
        val_loss = validate(model, val_loader, args.device)
        if val_loss < best_val_loss: 
            best_val_loss = val_loss
            model.save_pretrained("best_model_chunk")
            tokenizer.save_pretrained("best_model_chunk")
            print(f"New best model saved with validation loss: {val_loss:.3f}")
        print(f"Epoch {epoch+1}")
        print(f"Time: {timedelta(seconds=epoch_duration)}")
        print(f"Train Loss: {avg_loss:.3f}")
        print(f"Val Loss: {val_loss:.3f}")
        print(f"Val PPL: {torch.exp(torch.tensor(val_loss)):.3f}")
    total_time = time.time() - total_start_time
    print(f"\nTraining Complete!")
    print(f"Total Duration: {timedelta(seconds=total_time)}")
    print(f"Avg Epoch Time: {timedelta(seconds=sum(epoch_times)/len(epoch_times))}")

def validate(model, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            inputs = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=inputs, attention_mask=masks, labels=labels)
            total_loss += outputs.loss.item()
    return total_loss / len(loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--chunk_size", default=1024, type=int)
    parser.add_argument("--stride", default=512, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--normalization", type=str, default="raw",
                       choices=["raw", "normalized", "deduplicated"],
                       help="Text normalization level: raw, normalized, or deduplicated")
    parser.add_argument("--dropout_type", type=str, default="none",
                   choices=["none", "all", "ffn", "last_layer", "scaled_last"],
                   help="Dropout strategy: none|all|ffn|last_layer|scaled_last")
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    args = parser.parse_args()

    # Load and chunk dataset
    dataset = load_dataset("dogtooth/default_project_dev_test")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    
    dev_split = dataset['dev'].train_test_split(test_size=0.15)
    train_texts = [example['text'] for example in dev_split['train']]
    val_texts = [example['text'] for example in dev_split['test']]
    
    # Pass the text strings to ChunkedDataset
    train_dataset = ChunkedDataset(train_texts, tokenizer, 
                                 args.chunk_size, args.stride,normalization=args.normalization)
    val_dataset = ChunkedDataset(val_texts, tokenizer,
                                args.chunk_size, args.stride,normalization=args.normalization)

    collate_fn = partial(dynamic_collate_fn, pad_token_id=tokenizer.eos_token_id)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=4
    )

    # Load model with gradient checkpointing
    model = CustomGPT2.from_pretrained(args.model, use_cache=False)
    model.set_dropout_strategy(args.dropout_type, args.dropout_rate)
    model.modify_forward()  
    model.to(args.device)  
    model.gradient_checkpointing_enable()
    if args.fp16:
        print("Using mixed precision (FP16)")
    print(f"Training on {len(train_dataset)} chunks")
    train(model, train_loader, val_loader, args, tokenizer)

if __name__ == "__main__":
    start_time = time.time()
    main()
    total_time = time.time() - start_time
    print(f"Total Execution Time: {timedelta(seconds=total_time)}")