#!/usr/bin/env python3
"""
train_model_with_feats_fixed.py

Fixed feature-based Conv1D→LSTM streaming trainer:
 - 3-channel input: [normalized raw, envelope, RMS]
 - Per-window zero-mean/unit-variance normalization
 - Random gain augmentation (×0.8–1.2) on training only
 - Class weights computed from a fresh label stream
 - Automatic steps_per_epoch / validation_steps
"""

import os, math, numpy as np, tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, LSTM, TimeDistributed, Dense, AveragePooling1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from data_loader import get_dataset

# --- GPU setup (unchanged) ---
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
for g in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(g, True)

# --- Config ---
BATCH_SIZE     = 4
EPOCHS         = 40
SEQ_LEN        = 2000
TRAIN_MANIFEST = "train.txt"
VAL_MANIFEST   = "val.txt"
TEST_MANIFEST  = "test.txt"
CHECKPOINT     = "best_model_with_feats.h5"

# --- Preprocessing & augmentation ---
def preprocess_and_augment(sig, lbl):
    # 1) Per-window normalization
    mean = tf.reduce_mean(sig, axis=1, keepdims=True)
    std  = tf.math.reduce_std(sig,  axis=1, keepdims=True)
    sign = (sig - mean) / (std + 1e-6)

    # 2) Extract envelope & RMS from normalized signal
    rect = tf.abs(sign)
    env  = AveragePooling1D(pool_size=50, strides=1, padding='same')(rect)
    sq   = tf.square(sign)
    rms  = tf.sqrt(AveragePooling1D(pool_size=50, strides=1, padding='same')(sq) + 1e-6)

    # 3) Stack into 3-channel input
    feats = tf.concat([sign, env, rms], axis=-1)

    # 4) Random gain on train
    factor = tf.random.uniform([], 0.8, 1.2)
    feats = feats * factor

    return feats, lbl

def preprocess_only(sig, lbl):
    # Same as above but *no* random gain
    mean = tf.reduce_mean(sig, axis=1, keepdims=True)
    std  = tf.math.reduce_std(sig,  axis=1, keepdims=True)
    sign = (sig - mean) / (std + 1e-6)
    rect = tf.abs(sign)
    env  = AveragePooling1D(50,1,'same')(rect)
    sq   = tf.square(sign)
    rms  = tf.sqrt(AveragePooling1D(50,1,'same')(sq) + 1e-6)
    feats = tf.concat([sign, env, rms], axis=-1)
    return feats, lbl

# --- Model definition (unchanged) ---
def build_model():
    inp = Input(shape=(SEQ_LEN,3), name='emg_feats')
    x   = Conv1D(16, 21, padding='causal', activation='relu')(inp)
    x   = LSTM(32, return_sequences=True)(x)
    out = TimeDistributed(Dense(1, activation='sigmoid'))(x)
    return Model(inp, out)

# --- Helper to count lines in a manifest ---
def count_lines(fn):
    with open(fn) as f:
        return sum(1 for _ in f)

# --- Main training flow ---
def main():
    # 1) Compute class weights from a fresh, unbatched label stream
    label_ds = get_dataset(TRAIN_MANIFEST, batch_size=1, shuffle=False)
    label_ds = label_ds.map(lambda _,lbl: lbl, num_parallel_calls=tf.data.AUTOTUNE)
    label_ds = label_ds.unbatch().unbatch()  # flatten to scalar per-sample
    all_labels = np.array([int(x.numpy()) for x in label_ds])
    counts = np.bincount(all_labels, minlength=2)
    total  = all_labels.shape[0]
    class_weight = {i: total/counts[i] for i in range(len(counts))}
    print("Computed class weights ➞", class_weight)

    # 2) Build datasets
    train_ds = (
        get_dataset(TRAIN_MANIFEST, batch_size=BATCH_SIZE, shuffle=True)
        .map(preprocess_and_augment, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        get_dataset(VAL_MANIFEST, batch_size=BATCH_SIZE, shuffle=False)
        .map(preprocess_only, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        get_dataset(TEST_MANIFEST, batch_size=BATCH_SIZE, shuffle=False)
        .map(preprocess_only, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    # 3) Instantiate & compile
    model = build_model()
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    # 4) Callbacks
    ckpt  = ModelCheckpoint(CHECKPOINT, monitor='val_loss',
                            save_best_only=True, verbose=1)
    early = EarlyStopping(monitor='val_loss', patience=5,
                          verbose=1, restore_best_weights=True)

    # 5) Compute steps_per_* from manifest sizes
    steps_per_epoch    = math.ceil(count_lines(TRAIN_MANIFEST) / BATCH_SIZE)
    validation_steps   = math.ceil(count_lines(VAL_MANIFEST)   / BATCH_SIZE)

    # 6) Train!
    model.fit(
        train_ds,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        class_weight=class_weight,
        callbacks=[ckpt, early]
    )

    # 7) Final evaluation
    loss, acc = model.evaluate(test_ds)
    print(f"\nTest Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")

    model.save("final_model_with_feats_fixed.h5")
    print("✅ Done. Best weights in", CHECKPOINT)

if __name__=="__main__":
    main()
