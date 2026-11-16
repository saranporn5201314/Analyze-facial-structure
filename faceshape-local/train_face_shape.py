"""
เทรนโมเดลวิเคราะห์รูปทรงใบหน้า 7 แบบ: Oval, Round, Square, Heart, Diamond, Oblong, Pear
รวมข้อมูลจาก 2 dataset อัตโนมัติ (ดูที่มาใน README_SETUP.md)

รันแบบปกติด้วย: python train_face_shape.py
ก่อนรัน ให้ทำตาม README_SETUP.md ให้ครบ (ติดตั้ง package + ดาวน์โหลด dataset ทั้งสองชุด)
"""

import os
import json
import shutil
import random

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# ============================== CONFIG ==============================
CONFIG = {
    # โฟลเดอร์ raw dataset ทั้งสองชุด (ตามคำสั่ง kaggle datasets download ใน README_SETUP.md)
    "DATA_ROOTS": ["./data/dataset1", "./data/dataset2"],
    "FILTERED_DATA_ROOT": "./data_7class",
    "OUTPUT_DIR": "./output",
    # ชื่อคลาสต้องตรงกับชื่อโฟลเดอร์ใน dataset (ไม่สนตัวพิมพ์เล็ก/ใหญ่)
    "CLASSES": ["Diamond", "Heart", "Oblong", "Oval", "Pear", "Round", "Square"],
    "IMG_SIZE": 224,
    "BATCH_SIZE": 32,
    "QUICK_TEST": False,     # True = รันเร็วเพื่อเช็กว่าทุกอย่างทำงาน, False = เทรนเต็มรูปแบบ
    "QUICK_TEST_CAP": 60,   # จำนวนรูปสูงสุดต่อคลาสตอน QUICK_TEST (คลาสที่มีน้อยกว่านี้จะใช้เท่าที่มี)
    "SPLIT": {"train": 0.70, "val": 0.15, "test": 0.15},
    "SEED": 42,
}

IMG_EXTS = (".jpg", ".jpeg", ".png")


def find_class_images(data_roots, class_name):
    """ค้นหารูปทั้งหมดของคลาสนี้ในทุก dataset โดยหาโฟลเดอร์ที่ชื่อตรงกับคลาส (ไม่สนตัวพิมพ์เล็ก/ใหญ่)"""
    class_lower = class_name.lower()
    found = []
    for root in data_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            if os.path.basename(dirpath).lower() == class_lower:
                for fn in filenames:
                    if fn.lower().endswith(IMG_EXTS):
                        found.append(os.path.join(dirpath, fn))
    return found


def is_valid_image(path):
    """เช็กว่าเปิดรูปได้จริง (dataset มีบางไฟล์เสีย/ไม่ครบ ซึ่งจะทำให้การเทรนพังกลางทาง)"""
    try:
        with Image.open(path) as im:
            im.load()
        return True
    except Exception:
        return False


def build_dataset_splits(cfg):
    """รวมรูปจากทุก dataset ต่อคลาส แล้วแบ่ง train/val/test เอง เพราะ dataset แต่ละชุดมีโครงสร้างไม่เหมือนกัน"""
    random.seed(cfg["SEED"])
    counts = {}

    if os.path.isdir(cfg["FILTERED_DATA_ROOT"]):
        shutil.rmtree(cfg["FILTERED_DATA_ROOT"])  # เริ่มใหม่ทุกครั้งกันข้อมูลเก่าปนเปื้อน

    for cls in cfg["CLASSES"]:
        images = find_class_images(cfg["DATA_ROOTS"], cls)
        if not images:
            raise FileNotFoundError(
                f"ไม่พบรูปของคลาส '{cls}' ในโฟลเดอร์ {cfg['DATA_ROOTS']}\n"
                "ตรวจสอบว่าดาวน์โหลด dataset ทั้งสองชุดตาม README_SETUP.md แล้ว"
            )
        valid = []
        for p in images:
            if is_valid_image(p):
                valid.append(p)
            else:
                print(f"  ข้ามรูปที่ไฟล์เสีย: {p}")
        images = valid
        images.sort()  # ให้ลำดับคงที่ก่อน shuffle เพื่อให้ SEED ได้ผลเหมือนเดิมทุกครั้ง
        random.shuffle(images)

        if cfg["QUICK_TEST"]:
            images = images[: min(cfg["QUICK_TEST_CAP"], len(images))]

        n = len(images)
        n_train = int(n * cfg["SPLIT"]["train"])
        n_val = int(n * cfg["SPLIT"]["val"])
        # ส่วนที่เหลือทั้งหมดไปเป็น test กันพลาดปัดเศษ
        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        for split_name, files in splits.items():
            dst_dir = os.path.join(cfg["FILTERED_DATA_ROOT"], split_name, cls)
            os.makedirs(dst_dir, exist_ok=True)
            for src_path in files:
                # ใส่ prefix กันชื่อไฟล์ชนกันระหว่าง dataset ต่างแหล่ง
                base = os.path.basename(src_path)
                dst_path = os.path.join(dst_dir, f"{len(os.listdir(dst_dir))}_{base}")
                shutil.copy(src_path, dst_path)

        counts[cls] = {k: len(v) for k, v in splits.items()}

    print("จำนวนรูปต่อคลาส (train / val / test):")
    for cls, c in counts.items():
        print(f"  {cls:10s} {c['train']:4d} / {c['val']:4d} / {c['test']:4d}")

    return counts


def build_generators(cfg):
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
    )
    eval_datagen = ImageDataGenerator(rescale=1. / 255)

    train_dir = os.path.join(cfg["FILTERED_DATA_ROOT"], "train")
    val_dir = os.path.join(cfg["FILTERED_DATA_ROOT"], "val")
    test_dir = os.path.join(cfg["FILTERED_DATA_ROOT"], "test")

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=(cfg["IMG_SIZE"], cfg["IMG_SIZE"]),
        batch_size=cfg["BATCH_SIZE"], class_mode="categorical", shuffle=True,
    )
    val_gen = eval_datagen.flow_from_directory(
        val_dir, target_size=(cfg["IMG_SIZE"], cfg["IMG_SIZE"]),
        batch_size=cfg["BATCH_SIZE"], class_mode="categorical", shuffle=False,
    )
    test_gen = eval_datagen.flow_from_directory(
        test_dir, target_size=(cfg["IMG_SIZE"], cfg["IMG_SIZE"]),
        batch_size=cfg["BATCH_SIZE"], class_mode="categorical", shuffle=False,
    )
    return train_gen, val_gen, test_gen


def compute_weights(train_gen):
    """คำนวณ class weight เพื่อชดเชยคลาสที่มีรูปน้อย (Diamond, Pear)"""
    classes = np.unique(train_gen.classes)
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=train_gen.classes
    )
    weight_dict = dict(zip(classes, weights))
    idx_to_name = {v: k for k, v in train_gen.class_indices.items()}
    print("Class weight (ยิ่งรูปน้อย ยิ่งน้ำหนักเยอะ):")
    for idx, w in weight_dict.items():
        print(f"  {idx_to_name[idx]:10s} {w:.2f}")
    return weight_dict


def build_model(cfg):
    base_model = MobileNetV2(
        input_shape=(cfg["IMG_SIZE"], cfg["IMG_SIZE"], 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(len(cfg["CLASSES"]), activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


def train(cfg):
    os.makedirs(cfg["OUTPUT_DIR"], exist_ok=True)

    build_dataset_splits(cfg)
    train_gen, val_gen, test_gen = build_generators(cfg)

    class_indices_path = os.path.join(cfg["OUTPUT_DIR"], "class_indices.json")
    with open(class_indices_path, "w", encoding="utf-8") as f:
        json.dump(train_gen.class_indices, f, ensure_ascii=False, indent=2)
    print("Class indices:", train_gen.class_indices)

    class_weight = compute_weights(train_gen)

    model, base_model = build_model(cfg)
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=5, restore_best_weights=True
    )

    epochs_phase1 = 3 if cfg["QUICK_TEST"] else 20
    epochs_phase2 = 2 if cfg["QUICK_TEST"] else 12

    print("\n=== เฟส 1: เทรนเฉพาะ classification head ===")
    model.fit(
        train_gen, validation_data=val_gen, epochs=epochs_phase1,
        callbacks=[early_stop], class_weight=class_weight,
    )

    print("\n=== เฟส 2: Fine-tuning ===")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_gen, validation_data=val_gen, epochs=epochs_phase2,
        callbacks=[early_stop], class_weight=class_weight,
    )

    print("\n=== ประเมินผลบน test set ===")
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"Test accuracy: {test_acc:.4f}")

    y_pred = np.argmax(model.predict(test_gen), axis=1)
    y_true = test_gen.classes
    target_names = [name for name, _ in sorted(train_gen.class_indices.items(), key=lambda kv: kv[1])]
    print(classification_report(y_true, y_pred, target_names=target_names))
    print(confusion_matrix(y_true, y_pred))

    h5_path = os.path.join(cfg["OUTPUT_DIR"], "face_shape_model.h5")
    model.save(h5_path)
    print(f"\nบันทึกโมเดลที่ {h5_path}")

    return h5_path, class_indices_path


def export_tfjs(cfg, h5_path):
    import subprocess
    import sys

    tfjs_dir = os.path.join(cfg["OUTPUT_DIR"], "tfjs_model")
    os.makedirs(tfjs_dir, exist_ok=True)

    # ใช้ convert_to_tfjs.py แทน tensorflowjs_converter.exe (ดูเหตุผลในไฟล์นั้น)
    converter = os.path.join(os.path.dirname(os.path.abspath(__file__)), "convert_to_tfjs.py")
    cmd = [sys.executable, converter, h5_path, tfjs_dir]
    print("\n=== แปลงเป็น TensorFlow.js ===")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    class_indices_path = os.path.join(cfg["OUTPUT_DIR"], "class_indices.json")
    shutil.copy(class_indices_path, os.path.join(tfjs_dir, "class_indices.json"))

    print(f"\nเสร็จแล้ว! ไฟล์สำหรับใช้บนเว็บอยู่ที่: {tfjs_dir}")
    print("เอาไฟล์ทั้งหมดในโฟลเดอร์นี้ไปวางใน faceshape-web/model/")


if __name__ == "__main__":
    h5_path, _ = train(CONFIG)
    export_tfjs(CONFIG, h5_path)
