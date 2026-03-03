import argparse
import logging
from pathlib import Path

from src.fma import VariableFMADataset
from src.variants.vggish_model import VGGishFMA

# ----------------------------
# Setup logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] [%(asctime)s] [%(filename)s:%(lineno)d] %(message)s'
)

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Train or test FMA genre classification models")
    parser.add_argument("mode", choices=["train", "test"], help="Run mode: train or test")
    parser.add_argument("model", choices=["vggish"], help="Model to use")
    args = parser.parse_args()

    # ----------------------------
    # Load datasets
    # ----------------------------
    logging.info("[DATASET] Loading training dataset")
    train_ds = VariableFMADataset(split="training")

    logging.info("[DATASET] Loading validation dataset")
    val_ds = VariableFMADataset(split="validation", genre_encoder=train_ds.genre_encoder)

    logging.info("[DATASET] Loading test dataset")
    test_ds = VariableFMADataset(split="test", genre_encoder=train_ds.genre_encoder)

    # ----------------------------
    # Instantiate model
    # ----------------------------
    if args.model == "vggish":
        logging.info(f"[MAIN] Using model VGGishFMA")
        model = VGGishFMA()
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # ----------------------------
    # Train or test
    # ----------------------------
    if args.mode == "train":
        logging.info(f"[MAIN] Training model {args.model}")
        model.train_generic(train_dataset=train_ds, val_dataset=val_ds)
    elif args.mode == "test":
        logging.info(f"[MAIN] Testing model {args.model}")
        acc = model.test_generic(test_dataset=test_ds)
        logging.info(f"[MAIN] Test accuracy: {acc*100:.2f}%")
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()