import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.model_selection import StratifiedKFold
from transformers import *
import tokenizers
import os
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')


path = ""

tokenizer = tokenizers.ByteLevelBPETokenizer(
    vocab= path + '../config/vocab-roberta-base.json',
    merges = path + '../config/merges-roberta-base.txt',
    lowercase = True, #All tokens are in lower case
    add_prefix_space=True #Do not treat spaces like part of the tokens
)

#Get the ids to decode the neural network output
sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974} 

train_set = pd.read_csv(path+'../data/train.csv').fillna('')
train_set.head()

max_length = 96
ct = train_set.shape[0]
input_ids = np.ones((ct, max_length), dtype='int32')
attention_mask = np.zeros((ct, max_length), dtype='int32')
token_type_ids = np.zeros((ct, max_length),dtype='int32')
start_tokens = np.zeros((ct, max_length), dtype='int32')
end_tokens = np.zeros((ct, max_length), dtype='int32')

for i in range(ct):
    
    #Find where selected text sits inside the tweet
    text1 = " "+" ".join(train_set.loc[i, 'text'].split())
    text2 = " ".join(train_set.loc[i, 'selected_text'].split())
    idx = text1.find(text2)
    chars = np.zeros((len(text1)))
    chars[idx:idx+len(text2)] = 1
    if text1[idx-1] == ' ':
        chars[idx-1] = 1
        
    #Encode the text and find the selected_text offset, for the
    #encoded vector might not have the same length as the text vector
    enc = tokenizer.encode(text1)
    offsets = []
    idx = 0
    for t in enc.ids:
        w = tokenizer.decode([t])
        offsets.append((idx, idx+len(w)))
        idx += len(w)
    
    #Find the ids of the selected_text in the tokenized text
    tokens = []
    for k, (a, b) in enumerate(offsets):
        s = np.sum(chars[a:b])
        if s > 0:
            tokens.append(k)
    
    #After precessing, fill the vectors
    sent_token = sentiment_id[train_set.loc[i, 'sentiment']]
    input_ids[i, :len(enc.ids)+5] = [0] + enc.ids + [2,2] + [sent_token] + [2]
    attention_mask[i, :len(enc.ids)+5] = 1
    
    if len(tokens) > 0:
        start_tokens[i, tokens[0]+1]=1
        end_tokens[i, tokens[-1]+1] = 1


test_set = pd.read_csv(path+'../data/test.csv').fillna('')

ct = test_set.shape[0]
input_ids_test = np.ones((ct, max_length), dtype='int32')
attention_mask_test = np.zeros((ct, max_length), dtype='int32')
token_type_ids_test = np.zeros((ct, max_length),dtype='int32')

#We do not need to find the selected text for the testing set for
# the algorithm will detect and it will be compared with the fragment
# on the csv file
for i in range(ct):
    
    text1 = " " + " ".join(test_set.loc[i, 'text'].split())
    enc = tokenizer.encode(text1)
    sentiment_token = sentiment_id[test_set.loc[i, 'sentiment']]
    input_ids_test[i, :len(enc.ids)+5] = [0] + enc.ids + [2,2] + [sentiment_token] + [2]
    attention_mask_test[i, :len(enc.ids)+5] = 1
    
def build_model():
    ids = tf.keras.layers.Input((max_length,), dtype=tf.int32)
    att = tf.keras.layers.Input((max_length,), dtype=tf.int32)
    tok = tf.keras.layers.Input((max_length,), dtype=tf.int32)

    config = RobertaConfig.from_pretrained(path+'../config/config-roberta-base.json')
    bert_model = TFRobertaModel.from_pretrained(path+'../config/pretrained-roberta-base.h5',config=config)
    x = bert_model(ids,attention_mask=att,token_type_ids=tok)
    
    x1 = tf.keras.layers.Dropout(0.1)(x[0]) # dropout randomly sets input units to 0 with a frequency of 10%
    x1 = tf.keras.layers.Conv1D(filters=128, kernel_size=2, padding='same')(x1) # it creates kernel that is convolved with the layer input over a single spatial dimension
    x1 = tf.keras.layers.LeakyReLU()(x1) # it allows a small gradient when the unit is not active
    x1 = tf.keras.layers.Conv1D(filters=64, kernel_size=2, padding='same')(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)
    x1 = tf.keras.layers.Conv1D(filters=32, kernel_size=2, padding='same')(x1)
    x1 = tf.keras.layers.Dense(units=1)(x1) # just a regular densely-connected NN layer
    x1 = tf.keras.layers.LeakyReLU()(x1)
    x1 = tf.keras.layers.Flatten()(x1) # it flattens the input and does not affect the batch size
    x1 = tf.keras.layers.Activation('softmax')(x1) # it converts a real vector to a vector of categorical probabilities
    
    x2 = tf.keras.layers.Dropout(0.1)(x[0])
    x2 = tf.keras.layers.Conv1D(filters=128, kernel_size=2, padding='same')(x2)
    x2 = tf.keras.layers.LeakyReLU()(x2)
    x2 = tf.keras.layers.Conv1D(filters=64, kernel_size=2, padding='same')(x2)
    x2 = tf.keras.layers.LeakyReLU()(x2)
    x2 = tf.keras.layers.Conv1D(filters=32, kernel_size=2, padding='same')(x2)
    x2 = tf.keras.layers.Dense(units=1)(x2)
    x2 = tf.keras.layers.LeakyReLU()(x2)
    x2 = tf.keras.layers.Flatten()(x2)
    x2 = tf.keras.layers.Activation('softmax')(x2)

    model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1,x2])
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5) # Fine tuning
    model.compile(loss='categorical_crossentropy', optimizer=optimizer) # computes the crossentropy loss between the labels and predictions

    return model

model = build_model()
model.summary()

def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    if (len(a)==0) & (len(b)==0): return 0.5
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


jac = []; VER='v0'; DISPLAY=1 # USE display=1 FOR INTERACTIVE
oof_start = np.zeros((input_ids.shape[0], max_length))
oof_end = np.zeros((input_ids.shape[0], max_length))
preds_start = np.zeros((input_ids_test.shape[0], max_length))
preds_end = np.zeros((input_ids_test.shape[0], max_length))
skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=777)
for fold,(idxT,idxV) in enumerate(skf.split(input_ids,train_set.sentiment.values)):

    print('#'*25)
    print('### FOLD %i'%(fold+1))
    print('#'*25)
    
    K.clear_session()

    # callback to save the Keras model or model weights at some frequency 
    sv = tf.keras.callbacks.ModelCheckpoint(
        '%s-roberta-%i.h5'%(VER,fold), monitor='val_loss', verbose=1, save_best_only=True,
        save_weights_only=True, mode='auto', save_freq='epoch')

    # trains the model for a fixed number of epochs (iterations on a dataset)
    model.fit([input_ids[idxT,], attention_mask[idxT,], token_type_ids[idxT,]], [start_tokens[idxT,], end_tokens[idxT,]], 
        epochs=3, batch_size=32, verbose=DISPLAY, callbacks=[sv],
        validation_data=([input_ids[idxV,],attention_mask[idxV,],token_type_ids[idxV,]], 
        [start_tokens[idxV,], end_tokens[idxV,]]))
    
    # saves the model to Tensorflow SavedModel or a single HDF5 file
    print('Loading model...')
    model.load_weights('%s-roberta-%i.h5'%(VER,fold))
    
    
    # out-of-fold prediction
    print('Predicting OOF...')
    oof_start[idxV,],oof_end[idxV,] = model.predict([input_ids[idxV,],attention_mask[idxV,],token_type_ids[idxV,]], batch_size=8, verbose=DISPLAY)
    
    # generates output predictions for the input samples
    print('Predicting Test...')
    preds = model.predict([input_ids_test,attention_mask_test,token_type_ids_test], batch_size=8, verbose=DISPLAY)
    preds_start += preds[0]/skf.n_splits
    preds_end += preds[1]/skf.n_splits
    
    # display jaccard index found for each epoch
    all = []
    for k in idxV:
        start_index = np.argmax(oof_start[k,])
        end_index = np.argmax(oof_end[k,])
        if start_index > end_index: 
            selected_text = train_set.loc[k,'text'] # if the selected text is not found, we use the whole text 
        else:
            text_value = " "+" ".join(train_set.loc[k,'text'].split())
            encode_text = tokenizer.encode(text_value)
            selected_text = tokenizer.decode(encode_text.ids[a-1:b])
        all.append(jaccard(selected_text,train_set.loc[k,'selected_text']))
    jac.append(np.mean(all))
    print('>>>> FOLD %i Jaccard ='%(fold+1),np.mean(all)) # display jaccard index for the current fold
    print()
    
all = []
for k in range(input_ids_test.shape[0]):
    start_index = np.argmax(preds_start[k,])
    end_index = np.argmax(preds_end[k,])
    if start_index > end_index: 
        selected_text = test_set.loc[k,'text']
    else:
        text_value = " "+" ".join(test_set.loc[k,'text'].split())
        encode_text = tokenizer.encode(text_value)
        selected_text = tokenizer.decode(encode_text.ids[start_index-1:end_index])
    all.append(selected_text)

test_set['selected_text'] = all
pd.set_option('max_colwidth', 96)
test_set.sample(10)