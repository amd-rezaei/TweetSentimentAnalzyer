import numpy as np  
import tokenizers
import os 

TOKENIZER_PATH = os.getenv('TOKENIZER_PATH', 'config/vocab-roberta-base.json')
MERGES_PATH = os.getenv('MERGES_PATH', 'config/merges-roberta-base.txt')


# Load tokenizer and set sentiment ID mapping
tokenizer = tokenizers.ByteLevelBPETokenizer(
            TOKENIZER_PATH,
            MERGES_PATH,
            lowercase=True,
            add_prefix_space=True
        )
sentiment_id = {"positive": 1313, "negative": 2430, "neutral": 7974}

# Preprocess text for Triton input

def preprocess_text(text: str, sentiment: str, max_len=96):
    input_ids = np.ones((1, max_len), dtype='int32')
    attention_mask = np.zeros((1, max_len), dtype='int32')
    token_type_ids = np.zeros((1, max_len), dtype='int32')
    
    # Clean and encode the input text
    text = " " + " ".join(text.split())
    enc = tokenizer.encode(text)
    sentiment_token = sentiment_id.get(sentiment, sentiment_id["neutral"])
    
    available_length = max_len - 5  # Reserve space for special tokens
    token_ids = enc.ids[:available_length]
    
    input_ids[0, :len(token_ids) + 5] = [0] + token_ids + [2, 2] + [sentiment_token] + [2]
    attention_mask[0, :len(token_ids) + 5] = 1
    
    return input_ids, attention_mask, token_type_ids

# Post-process Triton logits to extract selected text
def get_selected_text_from_logits(text, start_logits, end_logits, max_len):
    encoded_text = tokenizer.encode(" " + " ".join(text.split()))
    start_idx = int(np.argmax(start_logits))
    end_idx = int(np.argmax(end_logits)) + 1  # Adjust end index to be inclusive

    if start_idx >= end_idx or end_idx > len(encoded_text.ids):
        return text.strip()
    else:
        selected_text = tokenizer.decode(encoded_text.ids[max(start_idx - 1, 0):end_idx]).strip()
        return selected_text