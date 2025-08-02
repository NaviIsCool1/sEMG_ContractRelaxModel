#!/usr/bin/env python3
"""
label_cleaned.py

Runs streaming inference and then suppresses tremor-induced false positives
by removing any detected "contract" bursts shorter than a specified duration.

Usage:
  python label_cleaned.py \
    --model  best_model_gpu_light.h5 \
    --input  raw_emg.csv \
    --output cleaned_labels.csv \
    --min_ms 150

Options:
  --min_ms    minimum burst duration in ms to keep (default: 150 ms)
"""

import argparse
import numpy as np
import tensorflow as tf
from collections import deque

def load_layers(model_path):
    model = tf.keras.models.load_model(model_path)
    conv      = model.get_layer('conv1d')
    lstm_cell = model.get_layer('lstm').cell
    dense     = model.get_layer('time_distributed').layer
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

        window   = tf.constant(np.array(buf)[None,:,:], dtype=tf.float32)
        conv_seq = conv(window)            
        conv_last= conv_seq[:, -1, :]      
        conv_out = tf.squeeze(conv_last, 0)

        inp = tf.expand_dims(conv_out, 0)
        h, [h, c] = lstm_cell(inp, states=[h, c])
        h = tf.squeeze(h, 0)

        p = dense(h[None,:])
        p = float(tf.squeeze(p).numpy())
        labels.append(int(p > 0.5))

    return np.array(labels, dtype=int)

def remove_short_bursts(labels, fs=2000, min_ms=150):
    min_len = int(fs * min_ms / 1000)
    out = labels.copy()
    N = len(labels)
    i = 0
    while i < N:
        if out[i] == 1:
            j = i
            while j < N and out[j] == 1:
                j += 1
            if (j - i) < min_len:
                out[i:j] = 0
            i = j
        else:
            i += 1
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model',   required=True, help='Path to .h5 model')
    p.add_argument('--input',   required=True, help='One-column CSV of mV readings')
    p.add_argument('--output',  required=True, help='CSV file for cleaned labels')
    p.add_argument('--min_ms',  type=float, default=150,
                   help='Min burst duration (ms) to keep as contraction')
    args = p.parse_args()

    samples = np.loadtxt(args.input, delimiter=',', dtype=np.float32)
    conv, lstm_cell, dense = load_layers(args.model)

    raw_labels = label_stream(conv, lstm_cell, dense, samples)
    cleaned = remove_short_bursts(raw_labels, fs=2000, min_ms=args.min_ms)

    np.savetxt(args.output, cleaned, fmt='%d')
    print(f"Wrote {len(cleaned)} cleaned labels to {args.output}")

if __name__ == "__main__":
    main()
