from tensorflow.contrib import skflow
from flask import current_app


def my_model(X, y):
    """This is DNN with 1000, 2000, 1000 hidden layers, and dropout of 0.01 probability."""
    layers = skflow.ops.dnn(X, [100, 1000, 2000, 1000, 100], dropout=0.01)
    return skflow.models.logistic_regression(layers, y)


def get_classifier():
	run_config = skflow.estimators.RunConfig(
        num_cores=current_app.config.get('NUM_CORES'),
        gpu_memory_fraction=current_app.config.get('GPU_MEMORY_FRACTION'))
	return skflow.TensorFlowEstimator(
		model_fn=my_model,
		n_classes=current_app.config.get('NUM_CLASSES'),
		steps=current_app.config.get('STEPS'),
		learning_rate=current_app.config.get('LEARNING_RATE'),
		config=run_config)
