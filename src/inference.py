import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import RobertaConfig, TFRobertaModel
import tokenizers

# Define constants
MAX_LEN = 96
PATH = '../config/'
weights_path = '../models/weights_final.h5'

# Load the tokenizer
vocab_pth = PATH + 'vocab-roberta-base.json'
merges_pth = PATH + 'merges-roberta-base.txt'
tokenizer = tokenizers.ByteLevelBPETokenizer(
    vocab_pth,
    merges_pth,
    lowercase=True,
    add_prefix_space=True
)

# Sentiment mapping
sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}

# Load test data
test = pd.read_csv('../data/test.csv').fillna('')

# Preprocess test data similar to the training process
ct = test.shape[0]
input_ids_t = np.ones((ct, MAX_LEN), dtype='int32')
attention_mask_t = np.zeros((ct, MAX_LEN), dtype='int32')
token_type_ids_t = np.zeros((ct, MAX_LEN), dtype='int32')

for k in range(test.shape[0]):
    # Tokenize input text
    text1 = " " + " ".join(test.loc[k, 'text'].split())
    enc = tokenizer.encode(text1)
    s_tok = sentiment_id[test.loc[k, 'sentiment']]
    input_ids_t[k, :len(enc.ids) + 5] = [0] + enc.ids + [2, 2] + [s_tok] + [2]
    attention_mask_t[k, :len(enc.ids) + 5] = 1

# Define model architecture (same as training)
def build_model():
    ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    tok = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

    config = RobertaConfig.from_pretrained(PATH + 'config-roberta-base.json')
    bert_model = TFRobertaModel.from_pretrained(PATH + 'pretrained-roberta-base.h5', config=config)
    
    x = bert_model(ids, attention_mask=att, token_type_ids=tok)
    
    x1 = tf.keras.layers.Dropout(0.1)(x[0])
    x1 = tf.keras.layers.Conv1D(1, 1)(x1)
    x1 = tf.keras.layers.Flatten()(x1)
    x1 = tf.keras.layers.Activation('softmax')(x1)
    
    x2 = tf.keras.layers.Dropout(0.1)(x[0])
    x2 = tf.keras.layers.Conv1D(1, 1)(x2)
    x2 = tf.keras.layers.Flatten()(x2)
    x2 = tf.keras.layers.Activation('softmax')(x2)

    model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1, x2])
    return model

# Load the model
model = build_model()
model.load_weights(weights_path)

# Make predictions on the test data
print("Predicting on test data...")
preds_start, preds_end = model.predict([input_ids_t, attention_mask_t, token_type_ids_t], verbose=1)

# Post-process predictions to extract selected text and save in required format
all_predictions = []
for k in range(input_ids_t.shape[0]):
    # Get the start and end token indices
    a = np.argmax(preds_start[k,])
    b = np.argmax(preds_end[k,])

    # Ensure indices are valid
    if a > b:
        st = test.loc[k, 'text']  # Use the original text if indices are invalid
    else:
        # Decode the tokens from the input text based on predicted indices
        text1 = " " + " ".join(test.loc[k, 'text'].split())
        enc = tokenizer.encode(text1)
        
        # Extend the selection slightly to avoid truncating the end of the text
        b = min(b + 1, len(enc.ids))  # Extend the end index by 1 token
        
        decoded_text = tokenizer.decode(enc.ids[max(a - 1, 0):b]).strip()
        st = decoded_text
    

    # Collect the relevant information for each test sample
    all_predictions.append({
        'textID': test.loc[k, 'textID'],
        'text': test.loc[k, 'text'],
        'selected_text': st,
        'sentiment': test.loc[k, 'sentiment']
    })

# Convert the list of predictions into a DataFrame
submission_df = pd.DataFrame(all_predictions)

# Save to CSV
submission_df.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created successfully.")
