# Updated train_model_gpu_light.py for improved generalization

#!/usr/bin/env python3
"""
train_model_gpu_light.py

GPU‐light training script with extra per‐window normalization and DC‐offset
augmentation to improve generalization across healthy & stroke cohorts.

Features added:
 - Zero‐mean, unit‐variance normalization per 1 s window
 - Random DC offset injection (±20 mV) during training
 - Maintains causal Conv1D→LSTM streaming architecture and GPU settings

Usage:
  (venv) $ python train_model_gpu_light.py
"""

import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, LSTM, TimeDistributed, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from data_loader import get_dataset

# GPU memory growth
for g in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(g, True)

# Config
BATCH_SIZE     = 4
EPOCHS         = 40
SEQ_LEN        = 2000
CHECKPOINT     = "best_model_gpu_light.h5"
TRAIN_MANIFEST = "train.txt"
VAL_MANIFEST   = "val.txt"
TEST_MANIFEST  = "test.txt"

def build_model():
    inp = Input(shape=(SEQ_LEN,1), name='emg_input')
    x = Conv1D(8, 21, padding='causal', activation='relu')(inp)
    x = LSTM(16, return_sequences=True)(x)
    out = TimeDistributed(Dense(1, activation='sigmoid'))(x)
    return Model(inputs=inp, outputs=out)

def normalize_and_augment(sig, lbl):
    # zero-mean, unit-variance
    sig = sig - tf.reduce_mean(sig, axis=1, keepdims=True)
    sig = sig / (tf.math.reduce_std(sig, axis=1, keepdims=True) + 1e-6)
    # inject random DC offset during training
    offset = tf.random.uniform([], -20.0, 20.0)
    sig = sig + offset
    return sig, lbl

def normalize_only(sig, lbl):
    sig = sig - tf.reduce_mean(sig, axis=1, keepdims=True)
    sig = sig / (tf.math.reduce_std(sig, axis=1, keepdims=True) + 1e-6)
    return sig, lbl

def main():
    # Load and preprocess datasets
    train_ds = (get_dataset(TRAIN_MANIFEST, BATCH_SIZE, shuffle=True)
                .map(normalize_and_augment, num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE))

    val_ds = (get_dataset(VAL_MANIFEST, BATCH_SIZE)
              .map(normalize_only, num_parallel_calls=tf.data.AUTOTUNE)
              .prefetch(tf.data.AUTOTUNE))

    test_ds = (get_dataset(TEST_MANIFEST, BATCH_SIZE)
               .map(normalize_only, num_parallel_calls=tf.data.AUTOTUNE)
               .prefetch(tf.data.AUTOTUNE))

    # Build & compile
    model = build_model()
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    # Callbacks
    ckpt  = ModelCheckpoint(CHECKPOINT, monitor='val_loss',
                            save_best_only=True, verbose=1)
    early = EarlyStopping(monitor='val_loss', patience=5,
                          verbose=1, restore_best_weights=True)

    # Train
    model.fit(train_ds,
              validation_data=val_ds,
              epochs=EPOCHS,
              callbacks=[ckpt, early])

    # Evaluate on test set
    loss, acc = model.evaluate(test_ds)
    print(f"\nTest Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")

    # Save final
    model.save("final_seq2seq_emg_gpu_light.h5")
    print("Done. Best weights in:", CHECKPOINT)

if __name__ == "__main__":
    main()


