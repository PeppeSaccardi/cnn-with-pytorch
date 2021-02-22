# config.py

SEED = 42
EXTENSION = ".png"
IMAGE_H = 28
IMAGE_W = 28
CHANNELS = 3

BATCH_SIZE = 30
EPOCHS = 400
LEARNING_RATE = 0.001

CIRCLES = "../input/shapes/circles/"
SQUARES = "../input/shapes/squares/"
TRIANGLES = "../input/shapes/triangles/"

INPUT_FOLD = "../input/"
OUTPUT_FOLD = "../output/"
TRAIN_DATA = "../input/train_dataset.csv" 
VALID_DATA = "../input/valid_dataset.csv"



ACCURACIES = "../output/accuracies.csv" 
LOSSES = "../output/losses.csv" 