import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import collections
import os
import datetime

from concurrent.futures import ThreadPoolExecutor, as_completed
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

dataset_directory = "C:\\DataScience\\Project\\dataset_livrable_1\\"
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

image_h = 180
image_w = 180
batch_s = 32

train_set, test_set = keras.utils.image_dataset_from_directory(
    dataset_directory,
    label_mode = "int",
    batch_size = batch_s,
    image_size = (image_h, image_w),
    seed = 42,
    validation_split = 0.2,
    subset = "both"
)

class_names = train_set.class_names
print(f"Classes détectées : {class_names}")

def print_class_distribution(dataset):
    label_counts = collections.Counter(label.numpy() for _, label in dataset.unbatch())

    classes = {0: "peintures", 1: "photos", 2: "schémas", 3: "croquis", 4: "textes scannés"}

    total = sum(label_counts.values())
    print(f"Nombre total d'images : {total}")

    for label, count in label_counts.items():
        class_name = classes.get(label, f"Classe inconnue ({label})")
        print(f"Nombre de {class_name} : {count} ({count / total * 100:.2f}% du total)")

print("------------------------- TRAIN SET -------------------------")
print_class_distribution(train_set)
print("------------------------- TEST SET --------------------------")
print_class_distribution(test_set)

images, labels = next(iter(train_set.take(1)))
print(f"Tensor des images : {images.shape}")
print(f"Tensor des labels : {labels.shape}")

plt.figure(figsize = (8, 8))
for images, labels in train_set.take(10):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i].numpy()])
        plt.axis("off")
    plt.savefig("sample_images.png")

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_set = train_set.cache().shuffle(1000).prefetch(buffer_size = AUTOTUNE)
test_set = test_set.cache().prefetch(buffer_size = AUTOTUNE)

num_classes = len(class_names)

def create_model(name, use_dropout = False, show_summary = True):
    model = Sequential(name = name)
    
    model.add(layers.Rescaling(1./255))
    model.add(layers.Conv2D(16, (3, 3), padding = 'same', activation = 'relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    if use_dropout:
        model.add(layers.Dropout(0.25))
    
    model.add(layers.Conv2D(32, (3, 3), padding = 'same', activation = 'relu'))
    model.add(layers.Conv2D(64, (3, 3), padding = 'same', activation = 'relu'))
    
    if use_dropout:
        model.add(layers.Dropout(0.25))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation = 'relu'))
    
    if use_dropout:
        model.add(layers.Dropout(0.5))
    
    model.add(layers.Dense(num_classes, activation = 'softmax'))
    
    model.compile(
        optimizer = 'adam',
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits = False),
        metrics = ['accuracy']
    )
    
    if show_summary:
        model.summary()
    
    return model

model = create_model("Base")

tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir = log_dir,
    histogram_freq = 1
)

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath = 'checkpoints/best_model.keras',
    monitor = 'val_accuracy',
    save_best_only = True,
    save_weights_only = False,
    mode = 'max',
    verbose = 1
)

callbacks = [tensorboard_callback, checkpoint_callback]

def train_model(model, train_set = train_set, test_set = test_set, epochs = 10):
    
    checkpoint_callback.filepath = f"checkpoints/best_{model.name.lower()}_model.keras"
    
    history = model.fit(
        train_set,
        validation_data = test_set,
        epochs = epochs,
        #callbacks = callbacks
    )
    
    accuracy = history.history['accuracy']
    validation_accuracy = history.history['val_accuracy']
    
    epochs_range = range(epochs) if epochs == 10 else range(len(accuracy))
    
    loss = history.history['loss']
    validation_loss = history.history['val_loss']
    
    plt.figure(figsize = (16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, accuracy, label = 'Training Accuracy')
    plt.plot(epochs_range, validation_accuracy, label = 'Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title(f"Training and Validation Accuracy - {model.name}")
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label = 'Training Loss')
    plt.plot(epochs_range, validation_loss, label = 'Validation Loss')
    plt.legend(loc='upper right')
    plt.title(f"Training and Validation Loss - {model.name}")
    
    plt.savefig("training_results.png")

train_model(model)

X_test = []
y_true = []

for images, labels in test_set:
    X_test.append(images)
    y_true.append(labels)

X_test = np.concatenate(X_test)
y_true = np.concatenate(y_true)

def display_matrix(model, X_test = X_test, y_true = y_true, class_names = class_names):
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis = 1)
    cm = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(cm, display_labels = class_names)
    display.plot(cmap = plt.cm.Blues)
    plt.title("Matrice de confusion")
    plt.xticks(rotation = 45)
    plt.savefig("confusion_matrix.png")

display_matrix(model)
