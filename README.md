# Brain_tumor_Classification
A deep learning classifier that analyzes brain MRI scans and predicts one of four categories:
**Glioma**, **Meningioma**, **Pituitary Tumor**, or **No Tumor**.

Built with a custom Convolutional Neural Network (CNN) trained on the
[Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).

---

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | **84.2%** |
| Training Images | 5,600 |
| Test Images | 1,600 |
| Classes | 4 (balanced) |
| Model | Custom CNN (3 conv blocks) |
| Input Size | 128×128 RGB |

### Per-Class Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Glioma | 0.89 | 0.78 | 0.83 |
| Meningioma | 0.97 | 0.55 | 0.70 |
| Notumor | 0.75 | 1.00 | 0.85 |
| Pituitary | 0.82 | 0.99 | 0.90 |

### Confusion Matrix

![Confusion Matrix](results/confusion_matrix.png)

### Training Curves

![Training Curves](results/training_curves.png)

---

## Approach

1. **Data Preprocessing** — Images resized to 128×128, pixel values normalized to [0, 1] via a `Rescaling` layer inside the model. Augmentation (rotation ±5%, horizontal flip, zoom ±5%) applied only to the training set.

2. **Model Architecture** — Custom CNN with 3 convolutional blocks (32, 64, 128 filters). Each block contains two `Conv2D` layers with ReLU activation, `BatchNormalization`, `MaxPooling2D`, and `Dropout(0.25)`. Followed by `GlobalAveragePooling2D`, `Dropout(0.4)`, `Dense(64)`, `Dropout(0.3)`, and a final `Dense(4, softmax)` output.

3. **Training** — Adam optimizer (learning rate 5e-4), categorical crossentropy loss, 30 epochs. Three callbacks: `EarlyStopping` (patience 8), `ReduceLROnPlateau` (factor 0.5, patience 4), and `ModelCheckpoint` (save best on val_accuracy).

4. **Evaluation** — Tested on 1,600 held-out images. Metrics: accuracy, precision, recall, F1-score, and confusion matrix.

---

## Project Structure

```
brain-tumor-classification/
├── src/
│   ├── __init__.py
│   ├── config.py           # Constants and paths
│   ├── data_loader.py      # tf.data pipeline
│   ├── model.py            # CNN architecture
│   ├── train.py            # Training loop with callbacks
│   └── evaluate.py         # Metrics and plots
├── results/
│   ├── confusion_matrix.png
│   └── training_curves.png
├── main.py                 # Training entry point
├── predict.py              # CLI prediction
├── app.py                  # Streamlit web app
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/brain-tumor-classification.git
cd brain-tumor-classification
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) from Kaggle.

Extract it so the structure looks like:

```
data/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

### 5. Train the model

```bash
python main.py
```

Training takes ~15 minutes on a T4 GPU (or ~2 hours on CPU).

---

## Usage

### Command-line prediction

```bash
python predict.py data/Testing/glioma/Te-gl_1.jpg
```

**Output:**
```
Prediction: glioma (89.39%)
All probabilities:
  glioma      :  89.39%
  meningioma  :   8.48%
  notumor     :   1.88%
  pituitary   :   0.24%
```

### Streamlit web app

```bash
python -m streamlit run app.py
```

Opens a browser interface at `http://localhost:8501` where you can upload an MRI image and see the prediction with confidence scores.

---

## Key Design Decisions

- **Custom CNN vs. transfer learning.** Transfer learning with ResNet50 was attempted but failed due to a TensorFlow 2.20 + Keras 3 environment issue with pretrained weight loading. A from-scratch CNN was the reliable fallback and achieved 84.2%.
- **128×128 input.** Chosen for faster training while retaining enough detail. Higher resolution (224×224) would add ~2-4% accuracy but triple training time.
- **Augmentation on training only.** Validation and test sets are not augmented to preserve honest evaluation.
- **Rescaling inside the model.** Ensures identical preprocessing at training and inference time.

---

## Limitations

- **Glioma/Meningioma confusion.** The model has 55% recall on meningioma, confusing it with pituitary and glioma in some cases. These classes look similar at 128×128 resolution.
- **Single dataset source.** Trained on one Kaggle dataset. Accuracy would drop 15-30% on MRIs from different scanners or protocols (domain shift).
- **Not clinically validated.** This is an educational proof-of-concept, not a medical diagnostic tool.

---

## Tech Stack

- **TensorFlow 2.20 / Keras 3** — Model building and training
- **NumPy** — Numerical operations
- **Matplotlib** — Plots
- **Scikit-learn** — Evaluation metrics
- **Streamlit** — Web interface
- **Pandas** — Data handling in the app

---

## Future Work

- **Transfer learning** — ResNet50/EfficientNet with pretrained weights (would likely reach 92-97%)
- **Higher resolution** — Train at 224×224 for finer detail
- **Grad-CAM** — Visualize which regions influenced each prediction
- **Cross-dataset validation** — Test on MRIs from other sources to quantify generalization
- **Tumor segmentation** — Extend from classification to pixel-level localization
