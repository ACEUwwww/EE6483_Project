import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

print(tf.config.list_physical_devices('GPU'))

IMAGE_SIZE: Tuple[int, int] = (150, 150)
BATCH_SIZE: int = 32
EPOCHS: int = 5
RANDOM_STATE: int = 42


def build_dataframe(image_dir: Path) -> pd.DataFrame:
    """Scan the directory and build a dataframe with file paths and labels."""
    if not image_dir.exists():
        raise FileNotFoundError(f"Directory not found: {image_dir}")

    filepaths = [
        path for path in image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    if not filepaths:
        raise FileNotFoundError(f"No image files found in {image_dir}.")

    records = []
    for path in sorted(filepaths):
        name = path.name.lower()
        if "cat" in name:
            label = "cat"
        elif "dog" in name:
            label = "dog"
        else:
            # Skip files without explicit cat/dog label in the filename
            continue
        records.append({"filepath": str(path), "label": label})

    if not records:
        raise ValueError(f"No filenames containing cat/dog labels found in {image_dir}.")

    df = pd.DataFrame(records)
    return df


def build_generators(train_df: pd.DataFrame):
    """Build training and validation data generators."""
    datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

    train_generator = datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="filepath",
        y_col="label",
        target_size=IMAGE_SIZE,
        class_mode="binary",
        batch_size=BATCH_SIZE,
        subset="training",
        shuffle=True,
        seed=RANDOM_STATE,
    )

    val_generator = datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="filepath",
        y_col="label",
        target_size=IMAGE_SIZE,
        class_mode="binary",
        batch_size=BATCH_SIZE,
        subset="validation",
        shuffle=False,
        seed=RANDOM_STATE,
    )

    return train_generator, val_generator


def build_test_generator(test_dir: Path):
    """Build test data generator and helper dataframe."""
    filepaths = [
        path for path in test_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    if not filepaths:
        raise FileNotFoundError(f"No image files found in {test_dir}.")

    try:
        filepaths = sorted(filepaths, key=lambda p: int(p.stem))
        ids = [int(p.stem) for p in filepaths]
    except ValueError:
        filepaths = sorted(filepaths, key=lambda p: p.name.lower())
        ids = [p.stem for p in filepaths]
    test_df = pd.DataFrame(
        {
            "filepath": [str(p) for p in filepaths],
            "id": ids,
        }
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col="filepath",
        y_col=None,
        target_size=IMAGE_SIZE,
        class_mode=None,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return test_generator, test_df


def build_model(input_shape: Tuple[int, int, int] = (150, 150, 3)) -> Sequential:
    """Create a simple convolutional neural network."""
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def evaluate_model(model: Sequential, val_generator, output_dir: Path):
    """Evaluate the model on the validation generator and save diagnostic plots."""
    print("Evaluating model on validation data...")
    val_generator.reset()
    y_prob = model.predict(val_generator).ravel()
    y_true = val_generator.classes
    y_pred = (y_prob >= 0.5).astype(int)

    class_indices = val_generator.class_indices
    class_names = [name for name, idx in sorted(class_indices.items(), key=lambda item: item[1])]

    cm = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
        print("Warning: ROC AUC could not be computed (only one class present).")
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )

    print("Validation classification report:")
    print(report)
    print(f"Validation F1 score: {f1:.4f}")
    print(f"Validation ROC AUC: {auc:.4f}")

    cm_fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    tick_marks = range(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    cm_fig.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    cm_fig.savefig(cm_path)
    plt.close(cm_fig)
    print(f"Confusion matrix saved to {cm_path}")

    if not np.isnan(auc):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.4f})")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        roc_fig.tight_layout()
        roc_path = output_dir / "roc_curve.png"
        roc_fig.savefig(roc_path)
        plt.close(roc_fig)
        print(f"ROC curve saved to {roc_path}")
    else:
        print("ROC curve was not generated.")


def parse_args():
    parser = argparse.ArgumentParser(description="Train or evaluate a cat vs dog CNN classifier.")
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Skip training and load an existing model.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to the saved model file (defaults to dog_cat_cnn.h5).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=f"Number of training epochs (default: {EPOCHS}).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    train_dir = project_root / "train"
    test_dir = project_root / "test1"
    epochs = args.epochs if args.epochs is not None else EPOCHS
    model_path = Path(args.model_path) if args.model_path else project_root / "dog_cat_cnn.h5"

    print("Indexing training data...")
    train_df = build_dataframe(train_dir)
    print(f"Number of training samples: {len(train_df)}")

    print("Building data generators...")
    train_gen, val_gen = build_generators(train_df)

    if args.no_train:
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        print(f"Loading model from {model_path}...")
        model = tf.keras.models.load_model(model_path)
    else:
        print("Building model...")
        model = build_model(input_shape=IMAGE_SIZE + (3,))
        print(f"Training model for {epochs} epochs...")
        model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            verbose=1,
        )
        model.save(model_path)
        print(f"Model saved to {model_path}")

    evaluate_model(model, val_gen, project_root)

    print("Preparing test generator and running inference...")
    test_gen, test_df = build_test_generator(test_dir)
    prob_predictions = model.predict(test_gen).ravel()
    binary_predictions = (prob_predictions >= 0.5).astype(int)

    submission = pd.DataFrame(
        {
            "id": test_df["id"],
            "label": binary_predictions,
        }
    )
    if submission["id"].dtype.kind in {"i", "u", "f"}:
        submission = submission.sort_values("id")
    submission_path = project_root / "submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Predictions written to {submission_path}")


if __name__ == "__main__":
    # Reduce TensorFlow logging verbosity
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    main()

