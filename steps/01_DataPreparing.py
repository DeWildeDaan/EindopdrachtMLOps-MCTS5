from ctypes import resize
from glob import glob
import json
import os
from datetime import datetime
import math
import random
import shutil

from utils import connectWithAzure

import cv2
from dotenv import load_dotenv
from azureml.core import Dataset
from azureml.data.datapath import DataPath


# When you work locally, you can use a .env file to store all your environment variables.
# This line read those in.
load_dotenv()

CLASSES = os.environ.get('CLASSES').split(',')
SEED = int(os.environ.get('RANDOM_SEED'))
IMAGE_SIZE = int(os.environ.get('IMAGE_SIZE'))
TRAIN_TEST_SPLIT_FACTOR = float(os.environ.get('TRAIN_TEST_SPLIT_FACTOR'))


def processAndUploadImages(datasets, data_path, processed_path, ws, class_name):

    # We can't use mount on these machines, so we'll have to download them

    image_path = os.path.join(data_path, 'classes', class_name)

    # Get the dataset name for this animal, then download to the directory
    # Overwriting means we don't have to delete if they already exist, in case something goes wrong.
    datasets[class_name].download(image_path, overwrite=True)
    print('Downloading all the images')

    # Get all the image paths with the `glob()` method.
    print(f'Resizing all images for {class_name} ...')
    # CHANGE THIS LINE IF YOU NEED TO GET YOUR ANIMAL_NAMES IN THERE IF NEEDED!
    image_paths = glob(f"{image_path}/*.jpg")

    # Process all the images with OpenCV. Reading them, then resizing them to 64x64 and saving them once more.
    print(f"Processing {len(image_paths)} images")
    for image_path in image_paths:
        image = cv2.imread(image_path)
        # Resize to a square of 64, 64
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        cv2.imwrite(os.path.join(processed_path, class_name,
                    image_path.split('/')[-1]), image)
    print(f'... done resizing. Stopping context now...')

    # Upload the directory as a new dataset
    print(f'Uploading directory now ...')
    resized_dataset = Dataset.File.upload_directory(
        # Enter the sourece directory on our machine where the resized pictures are
        src_dir=os.path.join(processed_path, class_name),
        # Create a DataPath reference where to store our images to. We'll use the default datastore for our workspace.
        target=DataPath(datastore=ws.get_default_datastore(),
                        path_on_datastore=f'processed_images/{class_name}'),
        overwrite=True)

    print('... uploaded images, now creating a dataset ...')

    # Make sure to register the dataset whenever everything is uploaded.
    new_dataset = resized_dataset.register(ws,
                                           name=f'resized_{class_name}',
                                           description=f'{class_name} images resized tot {IMAGE_SIZE}, {IMAGE_SIZE}',
                                           # Optional tags, can always be interesting to keep track of these!
                                           tags={
                                               'class': class_name, 'AI-Model': 'CNN', 'GIT-SHA': os.environ.get('GIT_SHA')},
                                           create_new_version=True)
    print(
        f" ... Dataset id {new_dataset.id} | Dataset version {new_dataset.version}")
    print(f'... Done. Now freeing the space by deleting all the images, both original and processed.')
    emptyDirectory(image_path)
    print(f'... done with the original images ...')
    emptyDirectory(os.path.join(processed_path, class_name))
    print(f'... done with the processed images. On to the next Animal, if there are still!')


def emptyDirectory(directory_path):
    shutil.rmtree(directory_path)


def prepareDataset(ws):
    data_folder = os.path.join(os.getcwd(), 'data')
    os.makedirs(data_folder, exist_ok=True)

    for class_name in CLASSES:
        os.makedirs(os.path.join(data_folder, 'classes',
                    class_name), exist_ok=True)

    # Define a path to store the animal images onto. We'll choose for `data/processed/animals` this time. Again, create subdirectories for all the animals
    processed_path = os.path.join(os.getcwd(), 'data', 'processed', 'classes')
    os.makedirs(processed_path, exist_ok=True)
    for class_name in CLASSES:
        os.makedirs(os.path.join(processed_path, class_name), exist_ok=True)

    # Make sure to give our workspace with it
    datasets = Dataset.get_all(workspace=ws)
    print(f"Available datasets:",datasets)
    for class_name in CLASSES:
        processAndUploadImages(
            datasets, data_folder, processed_path, ws, class_name)


def trainTestSplitData(ws):

    training_datapaths = []
    testing_datapaths = []
    default_datastore = ws.get_default_datastore()
    for class_name in CLASSES:
        # Get the dataset by name
        dataset = Dataset.get_by_name(ws, f"resized_{class_name}")
        print(f'Starting to process {class_name} images.')

        # Get only the .JPG images
        images = [
            img for img in dataset.to_path() if img.split('.')[-1] == 'jpg']

        print(f'... there are about {len(images)} images to process.')

        # Concatenate the names for the animal_name and the img_path. Don't put a / between, because the img_path already contains that
        # Make sure the paths are actual DataPaths
        images = [
            (default_datastore, f'processed_images/{class_name}{img_path}') for img_path in images]

        # Use the same random seed as I use and defined in the earlier cells
        random.seed(SEED)
        random.shuffle(images)  # Shuffle the data so it's randomized

        # Testing images
        # Get a small percentage of testing images
        amount_of_test_images = math.ceil(
            len(images) * TRAIN_TEST_SPLIT_FACTOR)

        test_images = images[:amount_of_test_images]
        training_images = images[amount_of_test_images:]

        # Add them all to the other ones
        testing_datapaths.extend(test_images)
        training_datapaths.extend(training_images)

        print(
            f'We already have {len(testing_datapaths)} testing images and {len(training_datapaths)} training images, on to process more images if necessary!')

    training_dataset = Dataset.File.from_files(path=training_datapaths)
    testing_dataset = Dataset.File.from_files(path=testing_datapaths)

    training_dataset = training_dataset.register(ws,
                                                 # Get from the environment
                                                 name=os.environ.get(
                                                     'TRAIN_SET_NAME'),
                                                 description=f'The Images to train, resized to {IMAGE_SIZE}, {IMAGE_SIZE}',
                                                 tags={'classes': os.environ.get('CLASSES'), 'AI-Model': 'CNN', 'Split size': str(
                                                     1 - TRAIN_TEST_SPLIT_FACTOR), 'type': 'training', 'GIT-SHA': os.environ.get('GIT_SHA')},
                                                 create_new_version=True)

    print(
        f"Training dataset registered: {training_dataset.id} -- {training_dataset.version}")

    testing_dataset = testing_dataset.register(ws,
                                               # Get from the environment
                                               name=os.environ.get(
                                                   'TEST_SET_NAME'),
                                               description=f'The Images to test, resized to {IMAGE_SIZE}, {IMAGE_SIZE}',
                                               tags={'classes': os.environ.get('CLASSES'), 'AI-Model': 'CNN', 'Split size': str(
                                                   TRAIN_TEST_SPLIT_FACTOR), 'type': 'testing', 'GIT-SHA': os.environ.get('GIT_SHA')},
                                               create_new_version=True)

    print(
        f"Testing dataset registered: {testing_dataset.id} -- {testing_dataset.version}")


def main():
    ws = connectWithAzure()

    print('Processing the images')
    prepareDataset(ws)

    print('Splitting the images')
    trainTestSplitData(ws)


if __name__ == '__main__':
    main()
