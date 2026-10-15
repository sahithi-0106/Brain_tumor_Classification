from src.data_loader import create_generators
from src.train import train_model
from src.evaluate import evaluate_model, plot_history

def main():
    print("=" * 50)
    print("Brain Tumor Classification")
    print("=" * 50)

    print("\n[1/4] Loading data...")
    train_ds, val_ds, test_ds = create_generators()

    print("\n[2/4] Training...")
    model, history = train_model(train_ds, val_ds)

    print("\n[3/4] Evaluating...")
    evaluate_model(model, test_ds)

    print("\n[4/4] Plotting...")
    plot_history(history)

    print("\nDone.")

if __name__ == "__main__":
    main()