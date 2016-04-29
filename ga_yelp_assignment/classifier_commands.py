import csv
import numpy as np
from collections import OrderedDict
from flask.ext.script import (Command, Manager, Option)
from sklearn import (cross_validation, metrics)
import tensorflow.contrib.learn
from tensorflow.contrib.learn import monitors
from ga_yelp_assignment.classifiers.cnn_classifier import get_classifier as cnn_classifier
from ga_yelp_assignment.classifiers.linear_classifier import get_classifier as l_classifier
from ga_yelp_assignment.classifiers.dnn_classifier import get_classifier as dnn_classifier
from ga_yelp_assignment.utils import (
    save_model,
    load_model,
    lazy_load_files,
    create_photo_label_dict,
    lazy_load_files)
from flask import current_app


available_classifiers = {
    'linear': l_classifier,
    'cnn': cnn_classifier,
    'dnn': dnn_classifier
}


def get_classifier(classifier, create_new, directory):
    if classifier not in available_classifiers:
            raise Exception(
                "Invalid classifier name supplied must be one of ({})".format(
                    ', '.join(available_classifiers)))
    if not create_new:
        return load_model(directory)
    else:
        return available_classifiers[classifier]()


def _validate_label(label):
    available_labels = [x.split('LABEL_')[1] for x in current_app.config.keys() if x.startswith('LABEL_')]
    if label not in available_labels:
        raise Exception(
            "Invalid label supplied, not one of ({})".format(', '.join(available_labels)))


def _validate_start_and_finish(start, finish):
    valid_range =[x for x in range(0, 110, 10)]
    err_msg = (
        "Start and finish parameters need to numbers between 0 and 100"
        " and multiple of 10. Finish parameter also needs to greater"
        " than start parameter")
    try:
        start = int(start)
        finish = int(finish)
    except:
        raise Exception(err_msg)

    if (start not in valid_range) or (finish not in valid_range) or (finish <= start):
        raise Exception(err_msg)
    return start, finish


def _split_train_test_dict(photo_label_dict, start=None, finish=None, test_size=0.8, random_state=5):
    # Seed the random state so we can process the list one by one.
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        list(photo_label_dict.keys()),
        list(photo_label_dict.values()),
        test_size=test_size,
        random_state=random_state)
    # # Cycle through the data 10% at a time
    if start is not None and finish is not None:
        chunk = int(len(X_train)/100)
        start = chunk * start if start != 0 else 0
        finish = chunk * finish
    else:
        start = 0
        finish = len(X_train)
    return OrderedDict(zip(X_train[start:finish], y_train[start:finish])), OrderedDict(zip(X_test, y_test))


class LoadTrainSaveClassifier(Command):

    option_list = (
        Option('--classifier', '-c', dest='classifier', help="The classifier to use."),
        Option('--label', '-l', dest='label', help="The label to train for"),
        Option('--dir', '-d', dest='directory', help="The directory to save to/load from the trained model"),
        Option('--new', '-n', dest='create_new', help="Create a new model"),
        Option('--start', '-s', dest='start', help="Start training from"),
        Option('--finish', '-f', dest='finish', help="Finish training to")
    )

    def run(self, classifier, label, directory, create_new, start, finish):
        new = True if create_new and create_new.lower().strip() == 'true' else False
        # # Validate the label
        _validate_label(label)
        # # Validate start and finish
        start, finish = _validate_start_and_finish(start, finish)
        # # Load the classifier
        clsf = get_classifier(classifier, new, directory)
        # Load the file names and true values
        img_name_dict = create_photo_label_dict(
            current_app.config.get('TRAIN_LABELS_CSV'),
            current_app.config.get('TRAIN_PHOTO_ID_CSV'),
            current_app.config.get('PHOTOS_DIRECTORY'),
            )
        # # Split the data up
        train_dict, test_dict = _split_train_test_dict(img_name_dict, start, finish)
        # Lazy load the images
        gen_ = lazy_load_files(OrderedDict(list(train_dict.items())), chunk_size=100)
        # Use first set of images as a validation monitor.
        val_images, val_values, file_names = next(gen_)
        val_monitor = monitors.ValidationMonitor(val_images, val_values, n_classes=10, print_steps=100)
        for images, values, file_names in gen_:
            clsf.continue_training = True
            # Train the classifier
            #clsf.fit(images, values, val_monitor, logdir=directory)
            clsf.fit(images, values, logdir=directory)

        clsf.continue_training = False
        # Save the classifier
        save_model(clsf, directory)


class LoadTestClassifier(Command):

    option_list = (
        Option('--classifier', '-c', dest='classifier'),
        Option('--dir', '-d', dest='directory'),
        Option('--label', '-l', dest='label'),
        Option('--out', '-o', dest='output')
    )

    def run(self, classifier, directory, label, output):
        # Validate the label
        _validate_label(label)
        # Load the classifier
        clsf = get_classifier(classifier, False, directory)
        # Load the file names and true values
        img_name_dict = create_photo_label_dict(
            current_app.config['TRAIN_LABELS_CSV'],
            current_app.config['TRAIN_PHOTO_ID_CSV'],
            current_app.config['PHOTOS_DIRECTORY'],
            )
        # Test the classifier
        train_dict, test_dict = _split_train_test_dict(img_name_dict)
        # Lazy load the images
        gen_ = lazy_load_files(OrderedDict(list(test_dict.items())), chunk_size=100)
        # Store results to a file
        with open(output, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['file_name', 'true', 'predicted'])
            # Store predicted versus actual
            pred_vals = []
            true_vals = []
            for images, values, file_names in gen_:
                pred_vals.extend(clsf.predict(images))
                true_vals.extend(values)
            # Store the files
            writer.writerows(zip(file_names, true_vals, pred_vals))


class CalculateLabelMetrics(Command):
    # http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics
    option_list = (
        Option('--in', '-i', dest='infile'),
    )

    def run(self, infile):
        pred_vals = None
        true_vals = None
        with open(infile, 'rU') as csvfile:
            reader = csv.reader(csvfile)
            rows = [x for x in reader]
            rows = rows[1:]
            file_names, pred_vals, true_vals = zip(*rows)

        pred_vals = np.array(pred_vals)
        true_vals = np.array(true_vals)
        print('F1: {0:f}'.format(metrics.f1_score(true_vals, pred_vals, pos_label='1')))
        print('Precision: {0:f}'.format(metrics.precision_score(true_vals, pred_vals, pos_label='1')))
        print('Recall: {0:f}'.format(metrics.recall_score(true_vals, pred_vals, pos_label='1')))
        print('Accuracy: {0:f}'.format(metrics.accuracy_score(true_vals, pred_vals)))
