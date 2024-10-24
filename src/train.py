import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd, numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.mixed_precision import LossScaleOptimizer, set_global_policy

# Enable mixed precision globally
set_global_policy('mixed_float16')

from sklearn.model_selection import StratifiedKFold
print('TF version',tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if len(tf.config.list_physical_devices('GPU')) == 0:
    os._exit(0)

from transformers import *
import tokenizers


MAX_LEN = 96
PATH = '../config/'

vocap_pth = PATH+'vocab-roberta-base.json'
merges_pth = PATH+'merges-roberta-base.txt'
    
tokenizer = tokenizers.ByteLevelBPETokenizer(
    lowercase=True,
    add_prefix_space=True
)
tokenizer.from_file(vocap_pth, merges_pth)
sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}
train = pd.read_csv('../data/train.csv').fillna('')
# train.head()


ct = train.shape[0]
input_ids = np.ones((ct,MAX_LEN),dtype='int32')
attention_mask = np.zeros((ct,MAX_LEN),dtype='int32')
token_type_ids = np.zeros((ct,MAX_LEN),dtype='int32')
start_tokens = np.zeros((ct,MAX_LEN),dtype='int32')
end_tokens = np.zeros((ct,MAX_LEN),dtype='int32')

for k in range(train.shape[0]):
    
    # FIND OVERLAP
    text1 = " "+" ".join(train.loc[k,'text'].split())
    text2 = " ".join(train.loc[k,'selected_text'].split())
    idx = text1.find(text2)
    chars = np.zeros((len(text1)))
    chars[idx:idx+len(text2)]=1
    if text1[idx-1]==' ': chars[idx-1] = 1 
    enc = tokenizer.encode(text1) 
        
    # ID_OFFSETS
    offsets = []; idx=0
    for t in enc.ids:
        w = tokenizer.decode([t])
        offsets.append((idx,idx+len(w)))
        idx += len(w)
    
    # START END TOKENS
    toks = []
    for i,(a,b) in enumerate(offsets):
        sm = np.sum(chars[a:b])
        if sm>0: toks.append(i) 
        
    s_tok = sentiment_id[train.loc[k,'sentiment']]
    input_ids[k,:len(enc.ids)+5] = [0] + enc.ids + [2,2] + [s_tok] + [2]
    attention_mask[k,:len(enc.ids)+5] = 1
    if len(toks)>0:
        start_tokens[k,toks[0]+1] = 1
        end_tokens[k,toks[-1]+1] = 1
        
        
test = pd.read_csv('../data/test.csv').fillna('')

ct = test.shape[0]
input_ids_t = np.ones((ct,MAX_LEN),dtype='int32')
attention_mask_t = np.zeros((ct,MAX_LEN),dtype='int32')
token_type_ids_t = np.zeros((ct,MAX_LEN),dtype='int32')

for k in range(test.shape[0]):
        
    # INPUT_IDS
    text1 = " "+" ".join(test.loc[k,'text'].split())
    enc = tokenizer.encode(text1)                
    s_tok = sentiment_id[test.loc[k,'sentiment']]
    input_ids_t[k,:len(enc.ids)+5] = [0] + enc.ids + [2,2] + [s_tok] + [2]
    attention_mask_t[k,:len(enc.ids)+5] = 1
    
def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    if (len(a) == 0) & (len(b) == 0): 
        return 0.5
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

import numpy as np
from sklearn.model_selection import StratifiedKFold
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.mixed_precision import LossScaleOptimizer, set_global_policy

# Enable mixed precision globally
set_global_policy('mixed_float16')

def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    if (len(a) == 0) & (len(b) == 0): 
        return 0.5
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def build_advanced_conv_model():
    ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    tok = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

    # Load pre-trained RoBERTa model configuration and weights
    config = RobertaConfig.from_pretrained(PATH + 'config-roberta-base.json')
    bert_model = TFRobertaModel.from_pretrained(PATH + 'pretrained-roberta-base.h5', config=config)
    
    # Freeze the pooler layer (since it's not needed for span prediction)
    for layer in bert_model.layers:
        if "pooler" in layer.name:
            layer.trainable = False

    x = bert_model(ids, attention_mask=att, token_type_ids=tok)

    # Start logits branch (using Conv1D and LeakyReLU)
    x_start = tf.keras.layers.Dropout(0.1)(x[0])
    x_start = tf.keras.layers.Conv1D(filters=128, kernel_size=2, padding='same')(x_start)
    x_start = tf.keras.layers.LeakyReLU()(x_start)
    x_start = tf.keras.layers.Conv1D(filters=64, kernel_size=2, padding='same')(x_start)
    x_start = tf.keras.layers.LeakyReLU()(x_start)
    x_start = tf.keras.layers.Conv1D(filters=32, kernel_size=2, padding='same')(x_start)
    x_start = tf.keras.layers.Dense(units=1)(x_start)
    x_start = tf.keras.layers.LeakyReLU()(x_start)
    x_start = tf.keras.layers.Flatten()(x_start)
    x_start = tf.keras.layers.Activation('softmax')(x_start)

    # End logits branch (using Conv1D and LeakyReLU)
    x_end = tf.keras.layers.Dropout(0.1)(x[0])
    x_end = tf.keras.layers.Conv1D(filters=128, kernel_size=2, padding='same')(x_end)
    x_end = tf.keras.layers.LeakyReLU()(x_end)
    x_end = tf.keras.layers.Conv1D(filters=64, kernel_size=2, padding='same')(x_end)
    x_end = tf.keras.layers.LeakyReLU()(x_end)
    x_end = tf.keras.layers.Conv1D(filters=32, kernel_size=2, padding='same')(x_end)
    x_end = tf.keras.layers.Dense(units=1)(x_end)
    x_end = tf.keras.layers.LeakyReLU()(x_end)
    x_end = tf.keras.layers.Flatten()(x_end)
    x_end = tf.keras.layers.Activation('softmax')(x_end)

    # Define model with two output branches: one for start logits and one for end logits
    model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x_start, x_end])

    # Use sparse_categorical_crossentropy for integer labels (token positions)
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    return model


# Main training loop with Stratified KFold cross-validation
jac = []
VER = 'v0'
DISPLAY = 1  # Enable verbose display
oof_start = np.zeros((input_ids.shape[0], MAX_LEN))
oof_end = np.zeros((input_ids.shape[0], MAX_LEN))
preds_start = np.zeros((input_ids_t.shape[0], MAX_LEN))
preds_end = np.zeros((input_ids_t.shape[0], MAX_LEN))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=777)

for fold, (idxT, idxV) in enumerate(skf.split(input_ids, train.sentiment.values)):

    print('#' * 25)
    print(f'### FOLD {fold + 1}')
    print('#' * 25)
    
    K.clear_session()
    model = build_advanced_conv_model()

    # Debug the shapes before training
    print("Training start_tokens shape:", start_tokens[idxT].shape)
    print("Training end_tokens shape:", end_tokens[idxT].shape)

    # Ensure start_tokens and end_tokens are integer arrays of shape (batch_size,)
    start_train = start_tokens[idxT] # Get the index positions of the start token
    end_train = end_tokens[idxT]      # Get the index positions of the end token
    start_valid = start_tokens[idxV]
    end_valid = end_tokens[idxV]

    # Print shapes for debugging
    print("Training start_train shape:", start_train.shape)
    print("Training end_train shape:", end_train.shape)
    
    # ModelCheckpoint callback to save best model during each fold
    sv = tf.keras.callbacks.ModelCheckpoint(
        '%s-roberta-%i.h5' % (VER, fold), monitor='val_loss', verbose=1, save_best_only=True,
        save_weights_only=True, mode='auto', save_freq='epoch'
    )
    
    # Training the model on the training set (idxT)
    print('Training model...')
    model.fit(
        [input_ids[idxT,], attention_mask[idxT,], token_type_ids[idxT,]],  # Input tensors
        [start_train, end_train],  # Integer token positions
        epochs=3,  # Increased epochs to 5
        batch_size=32,  # Keeping batch size as 32 for efficient training
        verbose=DISPLAY, callbacks=[sv],
        validation_data=([input_ids[idxV,], attention_mask[idxV,], token_type_ids[idxV,]], 
                         [start_valid, end_valid])  # Integer validation labels
    )
    
    # Load the best weights from training
    print('Loading best model weights...')
    model.load_weights('%s-roberta-%i.h5' % (VER, fold))
    
    # Predicting OOF for the current fold
    print('Predicting OOF...')
    oof_start[idxV,], oof_end[idxV,] = model.predict(
        [input_ids[idxV,], attention_mask[idxV,], token_type_ids[idxV,]], 
        verbose=DISPLAY, 
        batch_size=8
    )

    # Predicting on the test set
    print('Predicting Test...')
    preds = model.predict(
        [input_ids_t, attention_mask_t, token_type_ids_t], 
        verbose=DISPLAY, 
        batch_size=8 
    )
    preds_start += preds[0] / skf.n_splits
    preds_end += preds[1] / skf.n_splits

    # DISPLAY FOLD JACCARD
    all = []
    for k in idxV:
        a = np.argmax(oof_start[k,])
        b = np.argmax(oof_end[k,])

        # Check if indices are valid, else swap or fix bounds
        if a > b:
            a, b = b, a  # Swap indices if they are in the wrong order
        if b >= MAX_LEN:
            b = MAX_LEN - 1  # Clip the end index to the maximum allowed length
        
        text1 = " " + " ".join(train.loc[k, 'text'].split())
        enc = tokenizer.encode(text1)
        
        if a > 0 and b > 0:
            st = tokenizer.decode(enc.ids[a-1:b])
        else:
            st = train.loc[k, 'text']  # Default to full text if indices are invalid

        # Calculate Jaccard score
        jaccard_score = jaccard(st, train.loc[k, 'selected_text'])
        all.append(jaccard_score)

    fold_jaccard = np.mean(all)
    jac.append(fold_jaccard)
    print(f'>>>> FOLD {fold + 1} Jaccard =', fold_jaccard)
    print()
