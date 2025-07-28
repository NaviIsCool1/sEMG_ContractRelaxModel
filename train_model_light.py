# train_model_light.py

"""
train_model_light.py

A lighter version of train_model.py to fit within limited memory:
- Reduced batch size to lower memory footprint.
- Halved model size (16 Conv1D filters, 32 LSTM units).
- Still uses causal Conv1D → LSTM(return_sequences=True) for per-sample predictions.

Usage:
  (venv) $ python train_model_light.py
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, LSTM, TimeDistributed, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from data_loader import get_dataset

# Configurations
BATCH_SIZE     = 8    # reduced from 32 to fit memory
EPOCHS         = 40
SEQ_LEN        = 2000
CHECKPOINT     = "best_model_light.h5"
TRAIN_MANIFEST = "train.txt"
VAL_MANIFEST   = "val.txt"
TEST_MANIFEST  = "test.txt"

def build_model():
    inp = Input(shape=(SEQ_LEN, 1), name='emg_input')
    x = Conv1D(16, kernel_size=21, padding='causal', activation='relu', name='conv1d')(inp)  # 16 filters
    x = LSTM(32, return_sequences=True, name='lstm')(x)  # 32 units
    out = TimeDistributed(Dense(1, activation='sigmoid'), name='time_dist_dense')(x)
    return Model(inputs=inp, outputs=out, name='seq2seq_emg_light')

def main():
    # Load datasets with lower prefetch buffer
    train_ds = get_dataset(TRAIN_MANIFEST, batch_size=BATCH_SIZE, shuffle=True)
    val_ds   = get_dataset(VAL_MANIFEST,   batch_size=BATCH_SIZE)
    test_ds  = get_dataset(TEST_MANIFEST,  batch_size=BATCH_SIZE)

    # Build and compile model
    model = build_model()
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # Callbacks
    ckpt = ModelCheckpoint(CHECKPOINT, monitor='val_loss', save_best_only=True, verbose=1)
    early = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

    # Train
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[ckpt, early]
    )

    # Evaluate
    loss, acc = model.evaluate(test_ds)
    print(f"\nTest Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")

    # Save final model
    model.save("final_seq2seq_emg_light.h5")
    print("Training complete. Best weights in:", CHECKPOINT)

if __name__ == "__main__":
    main()

