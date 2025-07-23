#!/usr/bin/env python3
"""
Speech command CNN experiments

Compare different activation functions (tanh, relu, leaky_relu) on 
audio classification task. Runs experiments and saves results.

Usage:
    python model_experiments.py --dataset_dir /path/to/data --output_base results/
"""
import os
import json
import time
import argparse

import librosa
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

import tensorflow as tf
from tensorflow import keras


def count_samples(path):
    sizes = [len(os.listdir(os.path.join(path, d))) for d in os.listdir(path)]
    return pd.DataFrame(sizes, index=os.listdir(path), columns=['num_samples'])


def load_dataset(path):
    data, labels, samples = [], [], []
    for label in os.listdir(path):
        dir_ = os.path.join(path, label)
        for fname in os.listdir(dir_):
            y, sr = lr.load(os.path.join(dir_, fname), sr=16_000)
            data.append(y)
            samples.append(sr)
            labels.append(label)
    return data, labels, samples


def encode_labels(labels):
    code = {lab: i for i, lab in enumerate(sorted(set(labels)))}
    y = [code[lab] for lab in labels]
    return np.array(y), code


def build_model(layers, input_shape, num_classes):
    m = keras.Sequential()
    for i, L in enumerate(layers):
        t = L['type']
        if t == 'conv':
            kwargs = dict(filters=L['filters'], kernel_size=L['kernel_size'],
                          activation=L['activation'])
            if i == 0:
                m.add(keras.layers.Conv1D(input_shape=input_shape, **kwargs))
            else:
                m.add(keras.layers.Conv1D(**kwargs))
        elif t == 'pool':
            m.add(keras.layers.MaxPooling1D(pool_size=L['pool_size']))
        elif t == 'dropout':
            m.add(keras.layers.Dropout(rate=L['rate']))
        elif t == 'flatten':
            m.add(keras.layers.Flatten())
        elif t == 'dense':
            m.add(keras.layers.Dense(L['units'], activation=L['activation']))
    m.add(keras.layers.Dense(num_classes, activation='softmax'))
    m.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    return m


def train_and_evaluate_model(model, layers, X_train, y_train, X_test, y_test):
    print("Configuration:")
    for L in layers:
        c = L.copy()
        if 'activation' in c and callable(c['activation']):
            c['activation'] = c['activation'].__name__
        print(" ", c)
    start = time.time()
    history = model.fit(X_train, y_train, epochs=10,
                        validation_data=(X_test, y_test),
                        verbose=2)
    elapsed = time.time() - start

    y_pred = np.argmax(model.predict(X_test), axis=-1)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(
            y_test, y_pred, output_dict=True),
        'training_time': elapsed,
        'val_accuracy': history.history['val_accuracy'],
        'val_loss': history.history['val_loss']
    }
    return history, metrics


def plot_and_save_metrics(history, metrics, title, outdir):
    os.makedirs(outdir, exist_ok=True)
    fig, axs = plt.subplots(4, 1, figsize=(12, 24))

    # accuracy
    axs[0].plot(history.history['accuracy'], label='train')
    axs[0].plot(history.history['val_accuracy'], label='val')
    axs[0].set_title(f'{title} Accuracy'); axs[0].legend()

    # loss
    axs[1].plot(history.history['loss'], label='train')
    axs[1].plot(history.history['val_loss'], label='val')
    axs[1].set_title(f'{title} Loss'); axs[1].legend()

    # confusion matrix
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', ax=axs[2])
    axs[2].set_title(f'{title} Confusion Matrix')
    axs[2].set_xlabel('Pred'); axs[2].set_ylabel('True')

    # classification report
    rpt = pd.DataFrame(metrics['classification_report']).transpose()
    sns.heatmap(rpt.iloc[:-1, :-1], annot=True, fmt='.2f', ax=axs[3])
    axs[3].set_title(f'{title} Classification Report')

    fig.savefig(os.path.join(outdir, f'{title}_metrics.png'))
    plt.close(fig)

    # save history & metrics
    pd.DataFrame(history.history).to_csv(
        os.path.join(outdir, f'{title}_history.csv'), index=False)
    with open(os.path.join(outdir, f'{title}_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)


def generate_layer_configs(act_fn):
    depth = [
        [
            {'type': 'conv', 'filters': 8,  'kernel_size': 13, 'activation': act_fn},
            {'type': 'pool', 'pool_size': 3},
            {'type': 'dropout', 'rate': 0.3}
        ],
        [
            {'type': 'conv', 'filters': 8,  'kernel_size': 13, 'activation': act_fn},
            {'type': 'pool', 'pool_size': 3},
            {'type': 'dropout', 'rate': 0.3},
            {'type': 'conv', 'filters': 16, 'kernel_size': 11, 'activation': act_fn},
            {'type': 'pool', 'pool_size': 3},
            {'type': 'dropout', 'rate': 0.3}
        ],
        [
            {'type': 'conv', 'filters': 8,  'kernel_size': 13, 'activation': act_fn},
            {'type': 'pool', 'pool_size': 3},
            {'type': 'dropout', 'rate': 0.3},
            {'type': 'conv', 'filters': 16, 'kernel_size': 11, 'activation': act_fn},
            {'type': 'pool', 'pool_size': 3},
            {'type': 'dropout', 'rate': 0.3},
            {'type': 'conv', 'filters': 32, 'kernel_size': 9,  'activation': act_fn},
            {'type': 'pool', 'pool_size': 3},
            {'type': 'dropout', 'rate': 0.3}
        ],
    ]
    size = [
        [
            {'type': 'flatten'},
            {'type': 'dense', 'units': 128, 'activation': act_fn},
            {'type': 'dropout', 'rate': 0.3},
            {'type': 'dense', 'units': 64,  'activation': act_fn},
            {'type': 'dropout', 'rate': 0.3}
        ],
        [
            {'type': 'flatten'},
            {'type': 'dense', 'units': 256, 'activation': act_fn},
            {'type': 'dropout', 'rate': 0.3},
            {'type': 'dense', 'units': 128, 'activation': act_fn},
            {'type': 'dropout', 'rate': 0.3}
        ],
        [
            {'type': 'flatten'},
            {'type': 'dense', 'units': 512, 'activation': act_fn},
            {'type': 'dropout', 'rate': 0.3},
            {'type': 'dense', 'units': 256, 'activation': act_fn},
            {'type': 'dropout', 'rate': 0.3}
        ],
    ]
    filters = [
        [
            {'type': 'conv', 'filters': 16, 'kernel_size': 13, 'activation': act_fn},
            {'type': 'conv', 'filters': 32, 'kernel_size': 11, 'activation': act_fn}
        ]
    ]
    return depth, size, filters


def generate_combinations(depth_cfgs, size_cfgs, filter_cfgs):
    combos = []
    for d in depth_cfgs:
        for s in size_cfgs:
            for f in filter_cfgs:
                name = f"D{depth_cfgs.index(d)}_S{size_cfgs.index(s)}_F{filter_cfgs.index(f)}"
                combos.append({'name': name, 'layers': d + f + s})
    return combos


def evaluate_configs(configs, X_train, y_train, X_test, y_test, base_out):
    results = []
    for cfg in configs:
        od = os.path.join(base_out, cfg['name'])
        mp = os.path.join(od, f"{cfg['name']}_metrics.json")
        if os.path.exists(mp):
            print(f"Skipping {cfg['name']} (already done)")
            with open(mp) as f:
                m = json.load(f)
            results.append((cfg['name'], m))
            continue
        os.makedirs(od, exist_ok=True)
        model = build_model(cfg['layers'], input_shape=(16_000,1),
                            num_classes=len(np.unique(y_train)))
        hist, mets = train_and_evaluate_model(
            model, cfg['layers'], X_train, y_train, X_test, y_test)
        plot_and_save_metrics(hist, mets, cfg['name'], od)
        results.append((cfg['name'], mets))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', required=True,
                        help="Path to parent folder of class‑subfolders")
    parser.add_argument('--output_base', required=True,
                        help="Base path for all activation outputs")
    args = parser.parse_args()

    # load & prepare
    print("Counting samples:")
    print(count_samples(args.dataset_dir))
    data, labs, _ = load_dataset(args.dataset_dir)
    y, label_map = encode_labels(labs)
    X = np.array(data).reshape(-1, 16_000, 1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=44, shuffle=True)

    activations = {
        'Tanh':   tf.nn.tanh,
        'Relu':   tf.nn.relu,
        'Leaky':  tf.nn.leaky_relu
    }

    all_results = {}
    for name, fn in activations.items():
        print(f"\n=== Running {name} ===")
        depth_cfgs, size_cfgs, filter_cfgs = generate_layer_configs(fn)
        combos = generate_combinations(depth_cfgs, size_cfgs, filter_cfgs)
        outdir = os.path.join(args.output_base, name)
        res = evaluate_configs(combos, X_train, y_train, X_test, y_test, outdir)
        all_results[name] = res

    # save summary
    with open(os.path.join(args.output_base, 'all_activations_summary.json'), 'w') as f:
        json.dump(all_results, f, indent=4)
    print("\nAll experiments complete.")


if __name__ == '__main__':
    main() 