import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from src.model import build_model
from src.config import EPOCHS, MODEL_SAVE_PATH

def train_model(train_ds, val_ds):
    os.makedirs("models", exist_ok=True)
    model = build_model()

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=8,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4,
                          min_lr=1e-6, verbose=1),
        ModelCheckpoint(MODEL_SAVE_PATH, monitor="val_accuracy",
                        save_best_only=True, verbose=1)
    ]

    history = model.fit(train_ds, validation_data=val_ds,
                        epochs=EPOCHS, callbacks=callbacks)
    return model, history