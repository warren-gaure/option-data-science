import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import collections
import os
import datetime
import keras_tuner as kt


from concurrent.futures import ThreadPoolExecutor, as_completed
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

def create_model(name, use_dropout=False, show_summary=True, hparams=None,activation='relu'):
    model = Sequential(name=name)
    model.add(layers.Rescaling(1.0 / 255))

    # Use hyperparameters if available, otherwise default values
    num_units = hparams.get('units', 128) if hparams else 128
    activation = hparams.get('activation', 'relu') if hparams else 'relu'
    dropout_rate = hparams.get('dropout', 0.5) if hparams else 0.5

    model.add(layers.Conv2D(16, (3, 3), padding='same', activation=activation))
    model.add(layers.MaxPooling2D((2, 2)))

    if use_dropout and dropout_rate:
        model.add(layers.Dropout(dropout_rate))

    model.add(layers.Conv2D(32, (3, 3), padding='same', activation=activation))
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation=activation))

    if use_dropout and dropout_rate:
        model.add(layers.Dropout(dropout_rate))

    model.add(layers.Flatten())
    model.add(layers.Dense(num_units, activation=activation))

    if use_dropout and dropout_rate:
        model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(num_classes, activation='softmax'))

    learning_rate = hparams.get('lr', 0.001) if hparams else 0.001
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate) if hparams else 'adam'

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    if show_summary:
        model.summary()

    return model



################################################ 
# Train the model
################################################

callbacks = []

# Function to train model with optional hyperparameter tuning
def train_model(model, train_set, test_set, epochs=10, use_hyperparameters=False, tuner=None):
    if use_hyperparameters and tuner:
        train_size = int(0.8 * len(train_set))
        val_size = len(train_set) - train_size
        train_dataset = train_set.take(train_size)
        val_dataset = train_set.skip(train_size)

        tuner.search(train_dataset, validation_data=val_dataset, epochs=50, validation_split=0.2, callbacks=[stop_early])
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        model = tuner.hypermodel.build(best_hps)
        history = model.fit(train_dataset, validation_data=val_dataset, epochs=10, validation_split=0.2)
    else:
        history = model.fit(
            train_set,
            validation_data=test_set,
            epochs=epochs,
            callbacks=callbacks
        )
    
    accuracy = history.history['accuracy']
    validation_accuracy = history.history['val_accuracy']
    loss = history.history['loss']
    validation_loss = history.history['val_loss']
    
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(range(len(accuracy)), accuracy, label='Training Accuracy')
    plt.plot(range(len(accuracy)), validation_accuracy, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title(f"Training and Validation Accuracy - {model.name}")
    
    plt.subplot(1, 2, 2)
    plt.plot(range(len(loss)), loss, label='Training Loss')
    plt.plot(range(len(loss)), validation_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title(f"Training and Validation Loss - {model.name}")
    
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
# Hyperparameters
################################################

def build_model(hp):
    units = hp.Int("units", min_value=32, max_value=512, step=32)
    activation = hp.Choice("activation", ["relu", "tanh"])
    dropout = hp.Boolean("dropout")

    hparams = {
        "dense_units": units,
        "activation": activation,
        "dropout_3": 0.5 if dropout else 0.0 
    }

    model = create_model(
        name="Tuner",
        use_dropout=dropout,  
        show_summary=False,
        hparams=hparams
    )
    return model


tuner = kt.Hyperband(
    hypermodel = build_model,
    objective = 'val_accuracy',
    max_epochs = 10,
    factor = 3,
    directory = 'hyperband',
    project_name = 'hyperband_test'
)
stop_early = keras.callbacks.EarlyStopping(
    monitor = 'val_accuracy',
    patience = 5,
    restore_best_weights = True
)

################################################ 
# Execute the model
################################################

early_stopping = keras.callbacks.EarlyStopping(
    monitor = 'val_accuracy',
    patience = 2,
    mode = 'max',
    restore_best_weights = True
)

data_augmentation = keras.Sequential([
    layers.RandomFlip(input_shape = (image_h, image_w, 3), mode = 'horizontal_and_vertical'),
    layers.RandomRotation(factor = 0.1, fill_mode = 'nearest'),
    layers.RandomZoom(height_factor = 0.1, fill_mode = 'nearest'),
])

augmented_train_set = train_set.map(lambda x, y: (data_augmentation(x, training = True), y))

# Call the functions to create and train the model
use_hyperparameters = True
if use_hyperparameters:
    train_size = int(0.8 * len(train_set))  # 80% for training
    val_size = len(train_set) - train_size  # 20% for validation

    train_dataset = train_set.take(train_size)
    val_dataset = train_set.skip(train_size)

    tuner.search(train_dataset, validation_data=val_dataset, epochs=50, validation_split=0.2, callbacks=[stop_early])
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = create_model("HyperTuned_Model", use_dropout=True, hparams=best_hps.values)
else:
    model = create_model("All", use_dropout=True)

callbacks.append(early_stopping)
model = train_model(model, train_set=augmented_train_set, test_set=test_set, epochs=20, use_hyperparameters=use_hyperparameters, tuner=tuner)

display_matrix(model)
model.save("model.keras")