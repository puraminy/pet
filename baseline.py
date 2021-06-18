import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import logging
logging.basicConfig(level=logging.INFO)

import tensorflow_hub as hub
import tokenization
module_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
model_path = "/drive2/pretrained/bert/uncased_L-12_H-768_A-12/2"
#bert_layer = hub.KerasLayer(module_url, trainable=True)
bert_layer = hub.KerasLayer(model_path, trainable=True)

train=pd.read_table("data/atomic/atomic_train_nn_1k_per_prefix.tsv")
test=pd.read_table("data/atomic/atomic_validation_nn_100_per_prefix.tsv")

train['Review'] = (train['input_text'].map(str) +' '+ train['target_text']).apply(lambda row: str(row).strip())

test['Review'] = (test['input_text'].map(str) +' '+ test['target_text']).apply(lambda row: str(row).strip())

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []

    for text in texts:
        text = tokenizer.tokenize(text)

        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)

        tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len

        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

def build_model(bert_layer, max_len=512):
    input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    net = tf.keras.layers.Dense(64, activation='relu')(clf_output)
    net = tf.keras.layers.Dropout(0.2)(net)
    net = tf.keras.layers.Dense(32, activation='relu')(net)
    net = tf.keras.layers.Dropout(0.2)(net)
    out = tf.keras.layers.Dense(9, activation='softmax')(net)
    
    model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(tf.keras.optimizers.Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

max_len = 150
train_input = bert_encode(train.Review.values, tokenizer, max_len=max_len)
test_input = bert_encode(test.Review.values, tokenizer, max_len=max_len)
#train_labels = tf.keras.utils.to_categorical(train.prefix.values, num_classes=9)
train_labels = tf.keras.utils.to_categorical(train.prefix.factorize()[0], num_classes=9)

model = build_model(bert_layer, max_len=max_len)
model.summary()

checkpoint = tf.keras.callbacks.ModelCheckpoint('model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)

train_history = model.fit(
    train_input, train_labels, 
    validation_split=0.2,
    epochs=3,
    callbacks=[checkpoint, earlystopping],
    batch_size=32,
    verbose=1)

model.load_weights('model.h5')
test_pred = model.predict(test_input)
with open("output_bert/bert_predictions", "w") as f:
   print(test_pred, file= f)


