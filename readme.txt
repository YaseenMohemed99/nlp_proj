# Language Model Perplexity Evaluation

## Test Scripts

### base_test.py
- **Description**: Computes perplexity using full-text truncation to the model's maximum sequence length.
- **Method**:
  - Processes each document as a single input
  - Uses built-in truncation to handle long sequences
  - Suitable for models trained with standard truncation

### custom_test.py 
- **Description**: Computes perplexity using sliding window chunking.
- **Method**:
  - Splits documents into chunks (size=1024, stride=256)
  - Processes chunks in batches of 8
  - Uses masking to ignore padding tokens
  - Suitable for models trained with chunked data

## Training Approaches

### lm_finetune_text_dropout.py
- **Chunked Training**:
  - Uses sliding window chunks (size=1024, stride=256)
  - Supports text normalization/deduplication
  - Implements custom dropout strategies
  - Saved to best_model_chunk

### lm_finetune_text_truncate.py
- **Truncated Training**:
  - Uses standard truncation to model length
  - Simpler preprocessing
  - Saved to best_model

## Evaluation Commands

Run these from project root:

bash
# Chunk-trained model with chunked evaluation
python custom_test.py ./best_model_chunk

# Chunk-trained model with truncation evaluation
python base_test.py ./best_model_chunk  

# Trunc-trained model with truncation evaluation
python base_test.py ./best_model

# Trunc-trained model with chunked evaluation 
python custom_test.py ./best_model