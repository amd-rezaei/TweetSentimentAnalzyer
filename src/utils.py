import numpy as np  
import tokenizers
import os 

# Load tokenizer paths from environment variables
TOKENIZER_PATH = os.getenv('TOKENIZER_PATH', 'config/vocab-roberta-base.json')
MERGES_PATH = os.getenv('MERGES_PATH', 'config/merges-roberta-base.txt')

# Initialize tokenizer and sentiment ID mapping
tokenizer = tokenizers.ByteLevelBPETokenizer(
    vocab=TOKENIZER_PATH,
    merges=MERGES_PATH,
    lowercase=True,
    add_prefix_space=True
)
sentiment_id = {"positive": 1313, "negative": 2430, "neutral": 7974}

def preprocess_text(text: str, sentiment: str, max_len=96):
    """
    Preprocesses text and sentiment for model input.

    Parameters:
    - text (str): The input text for sentiment analysis.
    - sentiment (str): Sentiment of the text ('positive', 'negative', or 'neutral').
    - max_len (int): Maximum length for padding/truncation.

    Returns:
    - input_ids, attention_mask, token_type_ids (np.array): Arrays formatted for model input.
    """
    input_ids = np.ones((1, max_len), dtype='int32')
    attention_mask = np.zeros((1, max_len), dtype='int32')
    token_type_ids = np.zeros((1, max_len), dtype='int32')
    
    # Encode and clean the input text
    text = " " + " ".join(text.split())  # Ensures consistent spacing
    enc = tokenizer.encode(text)
    sentiment_token = sentiment_id.get(sentiment, sentiment_id["neutral"])  # Default to 'neutral' if sentiment is missing
    
    available_length = max_len - 5  # Reserve space for special tokens
    token_ids = enc.ids[:available_length]  # Truncate if necessary

    # Construct input sequences with special tokens
    input_ids[0, :len(token_ids) + 5] = [0] + token_ids + [2, 2] + [sentiment_token] + [2]
    attention_mask[0, :len(token_ids) + 5] = 1  # Mark valid tokens in attention mask
    
    return input_ids, attention_mask, token_type_ids

def get_selected_text_from_logits(text: str, start_logits: np.array, end_logits: np.array, max_len: int):
    """
    Post-processes model logits to extract the selected text span from the input text.

    Parameters:
    - text (str): Original input text.
    - start_logits (np.array): Logits for start positions.
    - end_logits (np.array): Logits for end positions.
    - max_len (int): Maximum tokenized length to consider.

    Returns:
    - selected_text (str): Extracted selected text based on start and end logits.
    """
    encoded_text = tokenizer.encode(" " + " ".join(text.split()))
    start_idx = int(np.argmax(start_logits))
    end_idx = int(np.argmax(end_logits)) + 1  # Adjust to make end index inclusive

    # Ensure indices are within bounds and make sense
    if start_idx >= end_idx or end_idx > len(encoded_text.ids):
        return text.strip()
    else:
        selected_text = tokenizer.decode(encoded_text.ids[max(start_idx - 1, 0):end_idx]).strip()
        return selected_text
