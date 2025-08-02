#!/usr/bin/env python3
"""
label_binary_gpu_light.py

Tweak of label_binary.py to work with the `train_model_gpu_light.py` model
(`best_model_gpu_light.h5`), whose TimeDistributed layer is named "time_distributed".

Usage:
  python label_binary_gpu_light.py \
    --model best_model_gpu_light.h5 \
    --input raw_emg.csv \
    --output labels.csv
"""

import argparse
import numpy as np
import tensorflow as tf
from collections import deque

def load_layers(model_path):
    model = tf.keras.models.load_model(model_path)
    conv = model.get_layer('conv1d')
    lstm_cell = model.get_layer('lstm').cell
    # For train_model_gpu_light.py, the TimeDistributed layer is named "time_distributed"
    dense = model.get_layer('time_distributed').layer
    return conv, lstm_cell, dense

def label_stream(conv, lstm_cell, dense, samples):
    K = conv.kernel_size[0]
    buf = deque(maxlen=K)
    U   = lstm_cell.units
    h   = tf.zeros((U,), dtype=tf.float32)
    c   = tf.zeros((U,), dtype=tf.float32)
    labels = []

    for x in samples:
        buf.append([x])
        if len(buf) < K:
            labels.append(0)
            continue

        # Build input tensor
        window = tf.constant(np.array(buf)[None,:,:], dtype=tf.float32)  # (1, K, 1)

        # Apply Conv1D and grab last time-step
        conv_seq  = conv(window)               # (1, K, F)
        conv_last = conv_seq[:, -1, :]         # (1, F)
        conv_out  = tf.squeeze(conv_last, 0)   # (F,)

        # LSTMCell step
        inp = tf.expand_dims(conv_out, 0)      # (1, F)
        h, [h, c] = lstm_cell(inp, states=[h, c])
        h = tf.squeeze(h, 0)                   # (U,)

        # Dense & threshold
        p = dense(h[None,:])                   # (1,1)
        p = float(tf.squeeze(p).numpy())
        labels.append(int(p > 0.5))

    return labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model',  required=True, help='Path to best_model_gpu_light.h5')
    p.add_argument('--input',  required=True, help='CSV of raw EMG (one mV per line)')
    p.add_argument('--output', required=True, help='CSV to write 0/1 labels')
    args = p.parse_args()

    samples = np.loadtxt(args.input, delimiter=',', dtype=np.float32)
    conv, lstm_cell, dense = load_layers(args.model)
    labels = label_stream(conv, lstm_cell, dense, samples)

    np.savetxt(args.output, labels, fmt='%d')
    print(f"Wrote {len(labels)} labels to {args.output}")

if __name__ == "__main__":
    main()

