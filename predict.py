import sys
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from src.config import IMG_SIZE, CLASS_NAMES

model = load_model("models/best_model.keras", compile=False)

def predict(path):
    img = image.load_img(path, target_size=IMG_SIZE)
    arr = image.img_to_array(img) 
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr, verbose=0)[0]
    top = int(np.argmax(preds))

    print(f"\nPrediction: {CLASS_NAMES[top]} ({preds[top]*100:.2f}%)")
    print("All probabilities:")
    for name, p in zip(CLASS_NAMES, preds):
        print(f"  {name:12s}: {p*100:6.2f}%")

if __name__ == "__main__":
    predict(sys.argv[1])