import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


IMAGE_SIZE: Tuple[int, int] = (150, 150)
BATCH_SIZE: int = 32
EPOCHS: int = 5
RANDOM_STATE: int = 42


def build_dataframe(image_dir: Path) -> pd.DataFrame:
    """扫描目录下的图片并生成文件路径与标签 DataFrame。"""
    if not image_dir.exists():
        raise FileNotFoundError(f"未找到目录：{image_dir}")

    filepaths = [
        path for path in image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    if not filepaths:
        raise FileNotFoundError(f"{image_dir} 中未发现图片文件。")

    records = []
    for path in sorted(filepaths):
        name = path.name.lower()
        if "cat" in name:
            label = "cat"
        elif "dog" in name:
            label = "dog"
        else:
            # 如果文件名不含 cat/dog，跳过该样本
            continue
        records.append({"filepath": str(path), "label": label})

    if not records:
        raise ValueError(f"{image_dir} 中未找到包含 cat/dog 标签的文件名。")

    df = pd.DataFrame(records)
    return df


def build_generators(train_df: pd.DataFrame):
    """构建训练与验证数据生成器。"""
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
    """构建测试集数据生成器与辅助 DataFrame。"""
    filepaths = [
        path for path in test_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    if not filepaths:
        raise FileNotFoundError(f"{test_dir} 中未发现图片文件。")

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
    """创建一个简单的卷积神经网络。"""
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

    print("构建训练数据索引...")
    train_df = build_dataframe(train_dir)
    print(f"训练样本数：{len(train_df)}")

    print("构建数据生成器...")
    train_gen, val_gen = build_generators(train_df)

    print("构建模型...")
    model = build_model(input_shape=IMAGE_SIZE + (3,))

    print("开始训练模型...")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        verbose=1,
    )

    model_path = project_root / "dog_cat_cnn.h5"
    model.save(model_path)
    print(f"模型已保存到 {model_path}")

    print("构建测试集生成器并进行推理...")
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
    print(f"预测结果已保存至 {submission_path}")


if __name__ == "__main__":
    # 限制 TensorFlow 的日志等级，保持输出简洁
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    main()

