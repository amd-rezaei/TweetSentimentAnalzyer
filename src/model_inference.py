# model_inference.py
import numpy as np
import tensorflow as tf
from transformers import RobertaConfig, TFRobertaModel
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from .utils import get_selected_text_from_logits, preprocess_text

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs detected: {[gpu.name for gpu in gpus]}")
else:
    print("No GPU detected!")

class TweetSentimentModel:
    def __init__(self, model_path, config_path, tokenizer_path, merges_path, weights_path, max_len=96):
        self.MAX_LEN = max_len
        self.model_path = model_path
        self.config_path = config_path
        self.weights_path = weights_path
        
        
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
    
    def convert_to_tensorrt(self, output_path):
        # Save the model temporarily in SavedModel format
        temp_model_dir = 'temp_model_dir'
        self.model.save(temp_model_dir)

        # Convert the model to TensorRT format
        converter = trt.TrtGraphConverterV2(input_saved_model_dir=temp_model_dir)
        converter.convert()
        
        # Save the converted TensorRT model
        converter.save(output_path)
        print(f"Model saved to {output_path} in TensorRT format")

    def preprocess(self, text, sentiment):
        input_ids, attention_mask, token_type_ids = preprocess_text(text, sentiment)

        return input_ids, attention_mask, token_type_ids

    def predict(self, text, sentiment):
        input_ids, attention_mask, token_type_ids = self.preprocess(text, sentiment)
        preds_start, preds_end = self.model.predict([input_ids, attention_mask, token_type_ids], verbose=0)
        
        selected_text = get_selected_text_from_logits(text, preds_start[0], preds_end[0], max_len=self.MAX_LEN)
        
        return selected_text
    
    def get_model(self):
        return self.model
