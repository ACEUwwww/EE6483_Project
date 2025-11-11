import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

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

    filepaths = sorted(filepaths, key=lambda p: p.name)
    test_df = pd.DataFrame(
        {
            "filepath": [str(p) for p in filepaths],
            "id": [p.stem for p in filepaths],
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


def main():
    project_root = Path(__file__).resolve().parent
    train_dir = project_root / "train"
    test_dir = project_root / "test1"

    print("Indexing training data...")
    train_df = build_dataframe(train_dir)
    print(f"Number of training samples: {len(train_df)}")

    print("Building data generators...")
    train_gen, val_gen = build_generators(train_df)

    print("Building model...")
    model = build_model(input_shape=IMAGE_SIZE + (3,))

    print("Training model...")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        verbose=1,
    )

    model_path = project_root / "dog_cat_cnn.h5"
    model.save(model_path)
    print(f"Model saved to {model_path}")

    print("Preparing test generator and running inference...")
    test_gen, test_df = build_test_generator(test_dir)
    predictions = model.predict(test_gen).ravel()

    submission = pd.DataFrame(
        {
            "id": test_df["id"],
            "label": predictions,
        }
    )
    submission_path = project_root / "submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Predictions written to {submission_path}")


if __name__ == "__main__":
    # Reduce TensorFlow logging verbosity
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    main()

