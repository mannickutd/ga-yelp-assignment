import os


class Config(object):
    # Flask config
    DEBUG = True
    # Data locations
    TRAIN_LABELS_CSV = os.environ['TRAIN_LABELS_CSV']
    TRAIN_PHOTO_ID_CSV = os.environ['TRAIN_PHOTO_ID_CSV']
    PHOTOS_DIRECTORY = os.environ['PHOTOS_DIRECTORY']
    # Data labels
    LABEL_0 = "GOOD_FOR_LUNCH"
    LABEL_1 = "GOOD_FOR_DINNER"
    LABEL_2 = "TAKES_RESERVATIONS"
    LABEL_3 = "OUTDOOR_SEATING"
    LABEL_4 = "RESTAURNAT_IS_EXPENSIVE"
    LABEL_5 = "HAS_ALCOHOL"
    LABEL_6 = "HAS_TABLE_SERVICE"
    LABEL_7 = "AMBIENCE_IS_CLASSY"
    LABEL_8 = "GOOD_FOR_KIDS"

    # Classifier variables
    IMAGE_X_DIM = 28
    IMAGE_Y_DIM = 28
    #CLASSIFIER_LOG_DIR = os.environ['CLASSIFIER_LOG_DIR']
    # Stochastic gradient variables
    NUM_CLASSES = 10
    LEARNING_RATE = 0.01
    STEPS = 100
    # Tensorflow configuration
    NUM_CORES = 3
    GPU_MEMORY_FRACTION = 0.6
