from tensorflow.contrib import skflow
from flask import current_app


def get_classifier():
	run_config = skflow.estimators.RunConfig(
        num_cores=current_app.config.get('NUM_CORES'),
        gpu_memory_fraction=current_app.config.get('GPU_MEMORY_FRACTION'))
	return skflow.TensorFlowLinearClassifier(
    	n_classes=current_app.config.get('NUM_CLASSES'),
    	steps=current_app.config.get('STEPS'),
    	learning_rate=current_app.config.get('LEARNING_RATE'),
    	config=run_config)
