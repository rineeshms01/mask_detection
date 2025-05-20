import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import classification_report, confusion_matrix

# Parameters
INIT_LR = 1e-4
EPOCHS = 20
BS = 32
IMAGE_SIZE = (224, 224)
MODEL_PATH = "mask_detector_model.keras"
LABEL_BINARIZER_PATH = "label_binarizer.pickle"
DATASET_DIR = "data"  # folder must contain 'with_mask' and 'without_mask' subfolders

# Data generators
train_aug = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2)

train_gen = train_aug.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BS,
    class_mode="categorical",
    subset="training")

val_gen = train_aug.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BS,
    class_mode="categorical",
    subset="validation")

# Save label map
label_map = train_gen.class_indices
with open(LABEL_BINARIZER_PATH, "wb") as f:
    pickle.dump(label_map, f)

# Load base model
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

# Freeze base model
for layer in baseModel.layers:
    layer.trainable = False

# Compile
opt = Adam(learning_rate=INIT_LR)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True)

# Train
H = model.fit(
    train_gen,
    steps_per_epoch=train_gen.samples // BS,
    validation_data=val_gen,
    validation_steps=val_gen.samples // BS,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint])

# Evaluate
val_gen.reset()
predIdxs = model.predict(val_gen, batch_size=BS)
predIdxs = np.argmax(predIdxs, axis=1)
true_labels = val_gen.classes
labels = list(label_map.keys())

print(classification_report(true_labels, predIdxs, target_names=labels))

# Confusion matrix
cm = confusion_matrix(true_labels, predIdxs)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# Training plot
plt.style.use("ggplot")
plt.figure()
plt.plot(H.history["loss"], label="Train Loss")
plt.plot(H.history["val_loss"], label="Val Loss")
plt.plot(H.history["accuracy"], label="Train Accuracy")
plt.plot(H.history["val_accuracy"], label="Val Accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig("training_plot.png")
plt.show()
