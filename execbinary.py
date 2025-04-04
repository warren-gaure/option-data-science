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
from sklearn.utils.class_weight import compute_class_weight

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
      
dataset_directory = "./dataset_livrable_1/"
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

image_h = 128
image_w = 128
batch_s = 16

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

################################################
# Create the model
################################################

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


################################################
# Train the model
################################################

callbacks = []
y_train = np.array([label.numpy() for _, label in train_set.unbatch()])
class_weights = compute_class_weight(class_weight = "balanced", classes = np.unique(y_train), y = y_train)
weights_dict = {cls: weight for cls, weight in zip(np.unique(y_train), class_weights)}

def train_model(model, train_set = train_set, test_set = test_set, epochs = 10, weights = weights_dict, use_hyperparameters = False, tuner = None):
    if use_hyperparameters and tuner:
        train_size = int(0.8 * len(train_set))
        #val_size = len(train_set) - train_size
        train_dataset = train_set.take(train_size)
        val_dataset = train_set.skip(train_size)

        tuner.search(
            train_dataset,
            validation_data = val_dataset,
            epochs = 50,
            validation_split = 0.2,
            callbacks = [stop_early]
        )
        
        best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
        model = tuner.hypermodel.build(best_hps)
        
        history = model.fit(
            train_dataset,
            validation_data = val_dataset,
            epochs = epochs,
            validation_split = 0.2
        )
    
    else:
        history = model.fit(
            train_set,
            validation_data = test_set,
            epochs = epochs,
            callbacks = callbacks,
            class_weight = weights
        )
    
    accuracy = history.history['accuracy']
    validation_accuracy = history.history['val_accuracy']
    
    loss = history.history['loss']
    validation_loss = history.history['val_loss']
    
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(range(len(accuracy)), accuracy, label = 'Training Accuracy')
    plt.plot(range(len(accuracy)), validation_accuracy, label = 'Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title(f"Training and Validation Accuracy")
    
    plt.subplot(1, 2, 2)
    plt.plot(range(len(loss)), loss, label = 'Training Loss')
    plt.plot(range(len(loss)), validation_loss, label = 'Validation Loss')
    plt.legend(loc='upper right')
    plt.title(f"Training and Validation Loss")
    
    plt.savefig("training_results.png")
    return model

################################################
# Display confusion matrix
################################################

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
  

################################################
# Execute classes
################################################


data_augmentation = keras.Sequential([
    layers.RandomFlip(input_shape = (image_h, image_w, 3), mode = 'horizontal_and_vertical'),
    layers.RandomRotation(factor = 0.1, fill_mode = 'nearest'),
    layers.RandomZoom(height_factor = 0.1, fill_mode = 'nearest'),
])
augmented_train_set = train_set.map(lambda x, y: (data_augmentation(x, training = True), y))

early_stopping = keras.callbacks.EarlyStopping(
    monitor = 'val_accuracy',
    patience = 2,
    mode = 'max',
    restore_best_weights = True
)

#model = create_model("Base")
#model = create_model("Augmentation")
#model = create_model("Dropout", use_dropout = True)
#model = create_model("Early Stopping")

#train_model(model)
#train_model(model, train_set = augmented_train_set)
#train_model(model, epochs = 20)

#display_matrix(model)

################################################
# Binary
################################################

train_set_binary, test_set_binary = keras.utils.image_dataset_from_directory(
    dataset_directory,
    label_mode = "int",
    batch_size = batch_s,
    image_size = (image_h, image_w),
    seed = 42,
    validation_split = 0.2,
    subset = "both"
)

photo_index = class_names.index("Photo")

def convert_label_to_binary(image, label):
    return image, tf.cast(tf.equal(label, photo_index), tf.int32)

train_set_binary = train_set_binary.map(convert_label_to_binary)
test_set_binary = test_set_binary.map(convert_label_to_binary)

binary_model = Sequential()

binary_model.add(layers.Rescaling(1./255))
binary_model.add(layers.Conv2D(16, (3, 3), padding = 'same', activation = 'relu'))
binary_model.add(layers.MaxPooling2D((2, 2)))
binary_model.add(layers.Dropout(0.25))
binary_model.add(layers.Conv2D(32, (3, 3), padding = 'same', activation = 'relu'))
binary_model.add(layers.Conv2D(64, (3, 3), padding = 'same', activation = 'relu'))
binary_model.add(layers.Dropout(0.25))
binary_model.add(layers.Flatten())
binary_model.add(layers.Dense(128, activation = 'relu'))
binary_model.add(layers.Dropout(0.5))
binary_model.add(layers.Dense(1, activation = 'sigmoid'))

binary_model.compile(
    optimizer = keras.optimizers.Adam(learning_rate = 0.001),
    loss = keras.losses.BinaryCrossentropy(from_logits = False),
    metrics = ['accuracy']
)

binary_weights = compute_class_weight(
    class_weight = "balanced",
    classes = np.array([0, 1]),
    y = np.array([label.numpy() for image, label in train_set_binary.unbatch()])
)

binary_weights_dict = {0: binary_weights[0], 1: binary_weights[1]}

X_test_binary = []
y_true_binary = []

for images, labels in test_set_binary:
    X_test_binary.append(images)
    y_true_binary.append(labels)

X_test_binary = np.concatenate(X_test_binary)
y_true_binary = np.concatenate(y_true_binary)

augmented_train_set_binary = train_set_binary.map(lambda x, y: (data_augmentation(x, training = True), y))

train_model(binary_model, train_set = augmented_train_set_binary, test_set = test_set_binary, weights = binary_weights_dict)      

display_matrix(binary_model, X_test = X_test_binary, y_true = y_true_binary, class_names = ['Non-photo', 'Photo'])