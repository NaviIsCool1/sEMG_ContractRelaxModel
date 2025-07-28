# train_model.py

"""
train_model.py

This script:
1. Loads the train, val, and test tf.data.Datasets from `data_loader.py`.
2. Defines a causal Conv1D + LSTM(return_sequences=True) model for per-sample sequence labeling:
   - Input: (batch, 2000, 1)
   - Conv1D layer with causal padding to learn MUAP-like filters.
   - LSTM layer (return_sequences=True) to capture temporal context per sample.
   - TimeDistributed Dense(sigmoid) head to output a probability at each time step.
3. Compiles with binary crossentropy and accuracy metrics.
4. Trains for a configurable number of epochs, with ModelCheckpoint to save the best weights.
5. Evaluates on the test set and saves the final model.

Usage:
  (venv) $ python train_model.py
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, LSTM, TimeDistributed, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from data_loader import get_dataset

# Configurations
BATCH_SIZE    = 32
EPOCHS        = 40
SEQ_LEN       = 2000
CHECKPOINT    = "best_model.h5"
TRAIN_MANIFEST= "train.txt"
VAL_MANIFEST  = "val.txt"
TEST_MANIFEST = "test.txt"

def build_model():
    inp = Input(shape=(SEQ_LEN, 1), name='emg_input')
    x = Conv1D(32, kernel_size=21, padding='causal', activation='relu', name='conv1d')(inp)
    x = LSTM(64, return_sequences=True, name='lstm')(x)
    out = TimeDistributed(Dense(1, activation='sigmoid'), name='time_dist_dense')(x)
    model = Model(inputs=inp, outputs=out, name='seq2seq_emg')
    return model

def main():
    # Load datasets
    train_ds = get_dataset(TRAIN_MANIFEST, batch_size=BATCH_SIZE, shuffle=True)
    val_ds   = get_dataset(VAL_MANIFEST, batch_size=BATCH_SIZE)
    test_ds  = get_dataset(TEST_MANIFEST, batch_size=BATCH_SIZE)

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
    model.save("final_seq2seq_emg_model.h5")
    print("Model training complete. Weights saved to:", CHECKPOINT)

if __name__ == "__main__":
    main()

