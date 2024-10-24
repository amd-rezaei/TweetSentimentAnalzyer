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
        self.id_to_sentiment = {0: 'positive', 1: 'negative', 2: 'neutral'}


        # Build and load model
        self.model = self.build_model()
        
        # Load the weights for the first 3 layers (start and end token prediction)
        self.model.load_weights(weights_path, by_name=True, skip_mismatch=True)

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

        # Add sentiment prediction layer (this is new)
        x3 = tf.keras.layers.GlobalAveragePooling1D()(x[0])  # Use global average pooling to reduce dimensions
        x3 = tf.keras.layers.Dense(3, activation='softmax', name='sentiment_output')(x3)  # 3 classes for sentiment

        model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1, x2, x3])
        return model

    def preprocess(self, text):
        # Prepare input for model prediction (only text, no sentiment input)
        input_ids = np.ones((1, self.MAX_LEN), dtype='int32')
        attention_mask = np.zeros((1, self.MAX_LEN), dtype='int32')
        token_type_ids = np.zeros((1, self.MAX_LEN), dtype='int32')

        # Tokenize text
        text = " " + " ".join(text.split())
        enc = self.tokenizer.encode(text)
        
        # For the sentiment placeholder, you can use the neutral sentiment token (e.g., ID 7974)
        neutral_sentiment_id = 7974  # Default to neutral sentiment for the placeholder
        input_ids[0, :len(enc.ids) + 5] = [0] + enc.ids + [2, 2] + [neutral_sentiment_id] + [2]
        attention_mask[0, :len(enc.ids) + 5] = 1

        return input_ids, attention_mask, token_type_ids


    def predict(self, text):
        # Preprocess input
        input_ids, attention_mask, token_type_ids = self.preprocess(text)

        # Predict start, end, and sentiment
        preds_start, preds_end, preds_sentiment = self.model.predict([input_ids, attention_mask, token_type_ids])

        # Get the most probable start and end positions
        a = np.argmax(preds_start[0,])
        b = np.argmax(preds_end[0,])

        # Ensure valid selection
        if a > b:
            selected_text = text.strip()
        else:
            enc = self.tokenizer.encode(" " + " ".join(text.split()))
            b = min(b + 1, len(enc.ids))
            selected_text = self.tokenizer.decode(enc.ids[max(a - 1, 0):b]).strip()

        # Get predicted sentiment class
        predicted_sentiment_id = np.argmax(preds_sentiment[0])
        predicted_sentiment = self.id_to_sentiment[predicted_sentiment_id]

        return selected_text, predicted_sentiment
