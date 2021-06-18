import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)

import tensorflow_hub as hub
import tokenization

def prepare_data(train_path, test_path):
    train = pd.read_table(train_path)
    test = pd.read_table(test_path)

    train["Review"] = (
        train["input_text"].map(str) + " " + train["target_text"]
    ).apply(lambda row: str(row).strip())

    test["Review"] = (
        test["input_text"].map(str) + " " + test["target_text"]
    ).apply(lambda row: str(row).strip())

    return train,test


def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []

    for text in texts:
        text = tokenizer.tokenize(text)

        text = text[: max_len - 2]
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
    input_word_ids = tf.keras.Input(
        shape=(max_len,), dtype=tf.int32, name="input_word_ids"
    )
    input_mask = tf.keras.Input(
        shape=(max_len,), dtype=tf.int32, name="input_mask"
    )
    segment_ids = tf.keras.Input(
        shape=(max_len,), dtype=tf.int32, name="segment_ids"
    )

    pooled_output, sequence_output = bert_layer(
        [input_word_ids, input_mask, segment_ids]
    )
    clf_output = sequence_output[:, 0, :]
    net = tf.keras.layers.Dense(64, activation="relu")(clf_output)
    net = tf.keras.layers.Dropout(0.2)(net)
    net = tf.keras.layers.Dense(32, activation="relu")(net)
    net = tf.keras.layers.Dropout(0.2)(net)
    out = tf.keras.layers.Dense(9, activation="softmax")(net)

    model = tf.keras.models.Model(
        inputs=[input_word_ids, input_mask, segment_ids], outputs=out
    )
    model.compile(
        tf.keras.optimizers.Adam(lr=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model

import click
@click.command()
@click.option(
    "--model_path",
    default = "/drive2/pretrained/bert/uncased_L-12_H-768_A-12/2",
    type=str,
    help=""
)
@click.option(
    "--train_path",
    default="data/atomic/atomic_train_nn_1k_per_prefix.tsv",
    type=str,
    help=""
)
@click.option(
    "--test_path",
    default="data/atomic/atomic_validation_nn_100_per_prefix.tsv",
    type=str,
    help=""
)
@click.option(
    "--do_train",
    "-t",
    is_flag=True,
    help=""
)
@click.option(
    "--do_eval",
    "-e",
    is_flag=True,
    help=""
)
@click.option(
    "--output_dir",
    default="output_bert",
    type=str,
    help=""
)
def train_eval(model_path, train_path, test_path, do_train, do_eval, output_dir):
    if not (do_train or do_eval):
        print("Please specify you want to train or eval with --do_train or --do_eval flags")
        return

    bert_layer = hub.KerasLayer(model_path, trainable=True)
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

    max_len = 150
    train, test = prepare_data(train_path, test_path)
    train_input = bert_encode(train.Review.values, tokenizer, max_len=max_len)
    test_input = bert_encode(test.Review.values, tokenizer, max_len=max_len)
    # train_labels = tf.keras.utils.to_categorical(train.prefix.values, num_classes=9)
    labels, uniques = train.prefix.factorize()
    train_labels = tf.keras.utils.to_categorical(
        labels, num_classes=9
    )

    model = build_model(bert_layer, max_len=max_len)

    model.summary()
    
    if do_train:
        print("=========== Training ... ")
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{output_dir}/checkpoints/best_model.h5",
            monitor="val_accuracy", save_best_only=True, verbose=1
        )
        earlystopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, verbose=1
        )

        train_history = model.fit(
            train_input,
            train_labels,
            validation_split=0.2,
            epochs=1,
            callbacks=[checkpoint, earlystopping],
            batch_size=16,
            verbose=1,
        )

    if do_eval:
        print("=========== Evaluating ... ")
        model.load_weights("output_bert/checkpoints/best_model.h5")
        y_prob = model.predict(test_input)
        pred_ids = y_prob.argmax(axis=-1)
        preds = uniques[pred_ids]
        with open(f"{output_dir}/bert_predictions", "w") as f:
            for pred in preds:
                print(pred, file=f)

        target_fname = f"{output_dir}/bert_targets"
        input_fname = f"{output_dir}/bert_inputs"
        if not Path(target_fname).is_file():
            with open(input_fname, "w") as inp_file:
                with open(target_fname, "w") as target_file:
                    for index, row in test.iterrows():
                        print(row["input_text"] + "--" + row["target_text"], file=inp_file)
                        print(row["prefix"], file=target_file)


if __name__ == "__main__":
    train_eval()
