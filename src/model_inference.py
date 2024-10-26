# model_inference.py
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import RobertaConfig, TFRobertaModel
import tokenizers

class TweetSentimentModel:
    def __init__(self, model_path, config_path, tokenizer_path, merges_path, weights_path, max_len=96):
        self.MAX_LEN = max_len
        self.model_path = model_path
        self.config_path = config_path
        self.weights_path = weights_path
        
        # Load tokenizer
        self.tokenizer = tokenizers.ByteLevelBPETokenizer(
            tokenizer_path,
            merges_path,
            lowercase=True,
            add_prefix_space=True
        )
        
        # Sentiment mapping
        self.sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}

        # Build and load model
        self.model = self.build_model()
        self.model.load_weights(weights_path, by_name=True, skip_mismatch=False)

    def build_model(self):
        ids = tf.keras.layers.Input((self.MAX_LEN,), dtype=tf.int32)
        att = tf.keras.layers.Input((self.MAX_LEN,), dtype=tf.int32)
        tok = tf.keras.layers.Input((self.MAX_LEN,), dtype=tf.int32)

        config = RobertaConfig.from_pretrained(self.config_path)
        bert_model = TFRobertaModel.from_pretrained(self.model_path, config=config)
        
        x = bert_model(ids, attention_mask=att, token_type_ids=tok)
        
        # Predict start and end positions for selected text
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
    
    def preprocess(self, text, sentiment):
        input_ids = np.ones((1, self.MAX_LEN), dtype='int32')
        attention_mask = np.zeros((1, self.MAX_LEN), dtype='int32')
        token_type_ids = np.zeros((1, self.MAX_LEN), dtype='int32')

        # Clean and encode the input text
        text = " " + " ".join(text.split())
        enc = self.tokenizer.encode(text)
        
        # Define the sentiment ID and calculate how much of enc.ids can be used
        sentiment_id = self.sentiment_id[sentiment]
        available_length = self.MAX_LEN - 5  # Reserve space for special tokens

        # Truncate enc.ids if it exceeds the available length
        token_ids = enc.ids[:available_length]

        # Construct the input with reserved special tokens and sentiment ID
        input_ids[0, :len(token_ids) + 5] = [0] + token_ids + [2, 2] + [sentiment_id] + [2]
        attention_mask[0, :len(token_ids) + 5] = 1

        return input_ids, attention_mask, token_type_ids

    def predict(self, text, sentiment):
        
        input_ids, attention_mask, token_type_ids = self.preprocess(text, sentiment)
        preds_start, preds_end = self.model.predict([input_ids, attention_mask, token_type_ids], verbose=0)
        

        a = np.argmax(preds_start[0])
        b = np.argmax(preds_end[0])

        if a > b:
            selected_text = text.strip()
        else:
            enc = self.tokenizer.encode(" " + " ".join(text.split()))
            b = min(b + 1, len(enc.ids))
            selected_text = self.tokenizer.decode(enc.ids[max(a - 1, 0):b]).strip()

        return selected_text
