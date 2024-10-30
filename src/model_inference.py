# model_inference.py
import numpy as np
import tensorflow as tf
from transformers import RobertaConfig, TFRobertaModel
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from .utils import get_selected_text_from_logits, preprocess_text

# Check for available GPUs
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

        # Sentiment label mapping to IDs
        self.sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}

        # Build and load the model with weights
        self.model = self.build_model()
        self.model.load_weights(self.weights_path, by_name=True, skip_mismatch=False)

    def build_model(self):
        """Builds and returns the model architecture based on Roberta."""
        # Define input layers
        ids = tf.keras.layers.Input(shape=(self.MAX_LEN,), dtype=tf.int32, name="input_ids")
        att = tf.keras.layers.Input(shape=(self.MAX_LEN,), dtype=tf.int32, name="attention_mask")
        tok = tf.keras.layers.Input(shape=(self.MAX_LEN,), dtype=tf.int32, name="token_type_ids")

        # Load model configuration and pretrained weights
        config = RobertaConfig.from_pretrained(self.config_path)
        bert_model = TFRobertaModel.from_pretrained(self.model_path, config=config)

        # Get BERT model outputs
        bert_output = bert_model(ids, attention_mask=att, token_type_ids=tok)

        # Start and end position prediction heads
        start_logits = tf.keras.layers.Dropout(0.1)(bert_output[0])
        start_logits = tf.keras.layers.Conv1D(1, 1)(start_logits)
        start_logits = tf.keras.layers.Flatten()(start_logits)
        start_logits = tf.keras.layers.Activation('softmax')(start_logits)

        end_logits = tf.keras.layers.Dropout(0.1)(bert_output[0])
        end_logits = tf.keras.layers.Conv1D(1, 1)(end_logits)
        end_logits = tf.keras.layers.Flatten()(end_logits)
        end_logits = tf.keras.layers.Activation('softmax')(end_logits)

        model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[start_logits, end_logits])
        return model

    def convert_to_tensorrt(self, output_path):
        """Converts the model to TensorRT format and saves it to output_path."""
        # Save the model temporarily in SavedModel format
        temp_model_dir = 'temp_model_dir'
        self.model.save(temp_model_dir)

        # Convert to TensorRT
        converter = trt.TrtGraphConverterV2(input_saved_model_dir=temp_model_dir)
        converter.convert()
        
        # Save the TensorRT model
        converter.save(output_path)
        print(f"Model successfully saved to {output_path} in TensorRT format")

    def preprocess(self, text, sentiment):
        """Preprocesses the input text and sentiment, returning model-compatible arrays."""
        input_ids, attention_mask, token_type_ids = preprocess_text(text, sentiment)
        return input_ids, attention_mask, token_type_ids

    def predict(self, text, sentiment):
        """Performs prediction on input text and sentiment, returning selected text."""
        input_ids, attention_mask, token_type_ids = self.preprocess(text, sentiment)
        preds_start, preds_end = self.model.predict([input_ids, attention_mask, token_type_ids], verbose=0)
        
        selected_text = get_selected_text_from_logits(
            text, preds_start[0], preds_end[0], max_len=self.MAX_LEN
        )
        
        return selected_text

    def get_model(self):
        """Returns the model instance."""
        return self.model
