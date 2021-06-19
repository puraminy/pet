import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import logging
from pathlib import Path
import os

logging.basicConfig(level=logging.INFO)

import tensorflow_hub as hub
import tokenization

ckp = 1
ckp_dir = ""
class CorrectCkpCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        #correct checkpoint number calculated from the base checkpoint number
        _from = epoch + 1 
        _to = ckp + epoch
        os.rename(f'{ckp_dir}/{_from:02d}__best_model.h5',f"{ckp_dir}/{_to:02d}_best_model.h5")


def prepare_data(train_file, test_file):
    train = pd.read_table(train_file)
    test = pd.read_table(test_file)

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
    "--train_file",
    default="data/atomic/atomic_train_nn_1k_per_prefix.tsv",
    type=str,
    help=""
)
@click.option(
    "--test_file",
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
    "--epochs",
    default=1,
    type=int,
    help="number of epochs"
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
@click.option(
    "--overwrite_output_dir",
    "-o",
    is_flag=True,
    help=""
)
@click.option(
    "--auto_output_dir",
    "-",
    is_flag=True,
    help=""
)
@click.option(
    "--do_train_eval",
    "-",
    is_flag=True,
    help=""
)
def train_eval(model_path, train_file, test_file, do_train, epochs, do_eval, output_dir, overwrite_output_dir, auto_output_dir, do_train_eval):
    
    global ckp, ckp_dir
    if do_train_eval:
        do_train = do_eval = True

    if not (do_train or do_eval):
        print("Please specify you want to train or eval with --do_train or --do_eval flags")
        return


    if Path(output_dir).is_file() and not overwrite_output_dir and not auto_output_dir:
        print(f"Output dir '{output_dir}' already exists, use --overwrite_output_dir if you want to overwrite it")
        return

    bert_layer = hub.KerasLayer(model_path, trainable=True)
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

    max_len = 150
    train, test = prepare_data(train_file, test_file)
    train_input = bert_encode(train.Review.values, tokenizer, max_len=max_len)
    test_input = bert_encode(test.Review.values, tokenizer, max_len=max_len)
    # train_labels = tf.keras.utils.to_categorical(train.prefix.values, num_classes=9)
    labels, uniques = train.prefix.factorize()
    train_labels = tf.keras.utils.to_categorical(
        labels, num_classes=9
    )

    if auto_output_dir:
        lt = len(train)
        lv = len(test)
        lv = "k".join(str(lv).rsplit("000", 1))
        lt = "k".join(str(lt).rsplit("000", 1))
        output_dir=f"output_bert_{lt}_{lv}"
        print("output dir:", output_dir)


    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ckp_dir = os.path.join(output_dir, "checkpoints") 
    Path(ckp_dir).mkdir(parents=True, exist_ok=True)
    ckp_file = os.path.join(ckp_dir, f"{ckp:02d}_best_model.h5")
    print("Checking if ", ckp_file, " exists...")
    last_ckp = ""
    while Path(ckp_file).is_file():
        print(ckp_file, " was found...")
        ckp += 1
        last_ckp = ckp_file
        ckp_file = os.path.join(ckp_dir, f"{ckp:02d}_best_model.h5")
        
    model = build_model(bert_layer, max_len=max_len)
    if last_ckp:
        print("Louding wheigts from ", last_ckp)
        model.load_weights(last_ckp)
        
    if do_train:
        #model.summary()
        print("=========== Training ... ")
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{ckp_dir}/" + "{epoch:02d}__best_model.h5",
            monitor="val_accuracy", save_best_only=True, verbose=1
        )
        earlystopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, verbose=1
        )

        train_history = model.fit(
            train_input,
            train_labels,
            validation_split=0.2,
            epochs=epochs,
            callbacks=[checkpoint, earlystopping, CorrectCkpCallback()],
            batch_size=16,
            verbose=1,
        )

    if do_eval:
        print("=========== Evaluating ... ")
        for i in range(1, ckp): 
            model_fname = f"{ckp_dir}/{i:02d}_best_model.h5"
            print("Evaluating ", model_fname)
            model.load_weights(model_fname)
            y_prob = model.predict(test_input)
            pred_ids = y_prob.argmax(axis=-1)
            preds = uniques[pred_ids]
            pred_fname = f"{output_dir}/bert_{i:02d}_predictions"
            if not Path(pred_fname).is_file():
                with open(pred_fname, "w") as f:
                    for pred in preds:
                        print(pred, file=f)

        target_fname = f"{output_dir}/bert_targets"
        input_fname = f"{output_dir}/bert_inputs"
        if not Path(target_fname).is_file():
            with open(input_fname, "w") as inp_file:
                with open(target_fname, "w") as target_file:
                    for index, row in test.iterrows():
                        print(str(row["input_text"]) + "--" + str(row["target_text"]), file=inp_file)
                        print(row["prefix"], file=target_file)


if __name__ == "__main__":
    train_eval()
