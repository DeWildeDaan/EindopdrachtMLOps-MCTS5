import argparse
import os
from glob import glob
import random
#import numpy as np

# This time we will need our Tensorflow Keras libraries, as we will be working with the AI training now
from tensorflow import keras
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# This AzureML package will allow to log our metrics etc.
from azureml.core import Run

# Important to load in the utils as well!
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--model-name', type=str, dest='model_name',
                    help='The name of the model to use.')

parser.add_argument('--training-folder', type=str,
                    dest='training_folder', help='training folder mounting point.')
parser.add_argument('--testing-folder', type=str,
                    dest='testing_folder', help='testing folder mounting point.')

parser.add_argument('--max-epochs', type=int, dest='max_epochs',
                    help='The maximum epochs to train.')
parser.add_argument('--seed', type=int, dest='seed',
                    help='The random seed to use.')
parser.add_argument('--initial-learning-rate', type=float,
                    dest='initial_lr', help='The initial learning rate to use.')
parser.add_argument('--batch-size', type=int, dest='batch_size',
                    help='The batch size to use during training.')
parser.add_argument('--patience', type=int, dest='patience',
                    help='The patience for the Early Stopping.')
parser.add_argument('--optimizer', type=str, dest='optimizer',
                    help='The optimizer of the model')
parser.add_argument('--image-size', type=int,
                    dest='image_size', help='The size of the image')


args = parser.parse_args()


training_folder = args.training_folder
print('Training folder:', training_folder)

testing_folder = args.testing_folder
print('Testing folder:', testing_folder)

MAX_EPOCHS = args.max_epochs  # Int
SEED = args.seed  # Int
INITIAL_LEARNING_RATE = args.initial_lr  # Float
BATCH_SIZE = args.batch_size  # Int
PATIENCE = args.patience  # Int
MODEL_NAME = args.model_name  # String
IMAGE_SIZE = (args.image_size, args.image_size, 3)  # Tuple
OPTIMIZER = args.optimizer  # String

CLASSES = ['infected', 'uninfected']


# As we're mounting the training_folder and testing_folder onto the `/mnt/data` directories, we can load in the images by using glob.
training_paths = glob(os.path.join('./data/train', '**',
                      'processed_images', '**', '*.jpg'), recursive=True)
testing_paths = glob(os.path.join('./data/test', '**',
                     'processed_images', '**', '*.jpg'), recursive=True)

print("Training samples:", len(training_paths))
print("Testing samples:", len(testing_paths))

print(training_paths[:3])  # Examples")
print(testing_paths[:3])  # Examples")

# Make sure to shuffle in the same way as I'm doing everything
random.seed(SEED)
random.shuffle(training_paths)
random.seed(SEED)
random.shuffle(testing_paths)

print(training_paths[:3])  # Examples
print(testing_paths[:3])  # Examples

# Parse to Features and Targets for both Training and Testing. Refer to the Utils package for more information
X_train = getFeatures(training_paths)
y_train = getTargets(training_paths)

X_test = getFeatures(testing_paths)
y_test = getTargets(testing_paths)

print('Shapes:')
print(X_train.shape)
print(X_test.shape)
print(len(y_train))
print(len(y_test))

print('Examples:' + str(y_test[0]))
print('Examples:' + str(y_train[0]))

# Make sure the data is one-hot-encoded
LABELS, y_train, y_test = encodeLabels(y_train, y_test)
print(LABELS)
print('One Hot Shapes:')

print(y_train.shape)
print(y_test.shape)

# Create an output directory where our AI model will be saved to.
# Everything inside the `outputs` directory will be logged and kept aside for later usage.
model_path = os.path.join('outputs', MODEL_NAME)
os.makedirs(model_path, exist_ok=True)

# START OUR RUN context.
# We can now log interesting information to Azure, by using these methods.
run = Run.get_context()

# Save the best model, not the last
cb_save_best_model = keras.callbacks.ModelCheckpoint(filepath=model_path,
                                                     monitor='val_loss',
                                                     save_best_only=True,
                                                     verbose=1)

# Early stop when the val_los isn't improving for PATIENCE epochs
cb_early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=PATIENCE,
                                              verbose=1,
                                              restore_best_weights=True)

# Reduce the Learning Rate when not learning more for 4 epochs.
cb_reduce_lr_on_plateau = keras.callbacks.ReduceLROnPlateau(
    factor=.5, patience=4, verbose=1)

# Create the AI model as defined in the utils script.
model = buildModel(IMAGE_SIZE, len(CLASSES))

model.compile(loss="categorical_crossentropy",
              optimizer=OPTIMIZER, metrics=["accuracy"])

# Add callback LogToAzure class to log to AzureML


class LogToAzure(keras.callbacks.Callback):
    '''Keras Callback for realtime logging to Azure'''

    def __init__(self, run):
        super(LogToAzure, self).__init__()
        self.run = run

    def on_epoch_end(self, epoch, logs=None):
        # Log all log data to Azure
        for k, v in logs.items():
            self.run.log(k, v)


# train the network
history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    steps_per_epoch=len(X_train) // BATCH_SIZE,
                    epochs=MAX_EPOCHS,
                    callbacks=[
                        LogToAzure(run),  # Thanks to Patrik De Boe!
                        cb_save_best_model,
                        cb_early_stop,
                        cb_reduce_lr_on_plateau
                    ])

print("[INFO] evaluating network...")
predictions = model.predict(X_test, batch_size=BATCH_SIZE)
# Give the target names to easier refer to them.
print(classification_report(y_test.argmax(axis=1),
      predictions.argmax(axis=1), target_names=CLASSES))
# If you want, you can enter the target names as a parameter as well, in case you ever adapt your AI model to more animals.

# log a final values
score = model.evaluate(X_test, y_test, verbose=0)
run.log("Final test loss", score[0])
print("Test loss:", score[0])

run.log("Final test accuracy", score[1])
print("Test accuracy:", score[1])

cf_matrix = confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))
print(cf_matrix)

print(LABELS)

# Log Confusion matrix , see https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.run.run?view=azure-ml-py#log-confusion-matrix-name--value--description----
cmtx = {
    "schema_type": "confusion_matrix",
    # "parameters": params,
    "data": {
        "class_labels": CLASSES,   # ["0", "1"]
        "matrix": [[int(y) for y in x] for x in cf_matrix]
    }
}

run.log_confusion_matrix('Confusion matrix - error rate', cmtx)

# Save the confusion matrix to the outputs.
np.save('outputs/confusion_matrix.npy', cf_matrix)

print("DONE TRAINING. AI model has been saved to the outputs.")
