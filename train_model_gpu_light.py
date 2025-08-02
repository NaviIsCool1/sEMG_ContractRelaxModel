# train_model_gpu_light.py

"""
train_model_gpu_light.py

A further-light GPU training script to avoid OOM on Colab Tesla T4:
- Sets TF_GPU_ALLOCATOR to 'cuda_malloc_async' to reduce fragmentation.
- Enables dynamic GPU memory growth.
- Reduces batch size to 4.
- Halves model capacity: 8 Conv1D filters, 16 LSTM units.
- Keeps causal Conv1D → LSTM(return_sequences=True) → TimeDistributed(sigmoid) architecture.

Usage:
  (venv) $ python train_model_gpu_light.py
"""

import os
# Set asynchronous GPU allocator to mitigate fragmentation
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, LSTM, TimeDistributed, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from data_loader import get_dataset

# Enable GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Configurations
BATCH_SIZE     = 4     # further reduced batch size
EPOCHS         = 40
SEQ_LEN        = 2000
CHECKPOINT     = "best_model_gpu_light.h5"
TRAIN_MANIFEST = "train.txt"
VAL_MANIFEST   = "val.txt"
TEST_MANIFEST  = "test.txt"

def build_model():
    inp = Input(shape=(SEQ_LEN, 1), name='emg_input')
    x = Conv1D(8, kernel_size=21, padding='causal', activation='relu')(inp)  # 8 filters
    x = LSTM(16, return_sequences=True)(x)  # 16 units
    out = TimeDistributed(Dense(1, activation='sigmoid'))(x)
    return Model(inputs=inp, outputs=out)

def main():
    # Load datasets
    train_ds = get_dataset(TRAIN_MANIFEST,  batch_size=BATCH_SIZE, shuffle=True)
    val_ds   = get_dataset(VAL_MANIFEST,    batch_size=BATCH_SIZE)
    test_ds  = get_dataset(TEST_MANIFEST,   batch_size=BATCH_SIZE)

    # Build & compile
    model = build_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # Callbacks
    ckpt  = ModelCheckpoint(CHECKPOINT, monitor='val_loss', save_best_only=True, verbose=1)
    early = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

    # Train
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[ckpt, early])

    # Evaluate
    loss, acc = model.evaluate(test_ds)
    print(f"\nTest Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")

    # Save final model
    model.save("final_seq2seq_emg_gpu_light.h5")
    print("Training complete. Best weights in:", CHECKPOINT)

if __name__ == "__main__":
    main()

