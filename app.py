from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import os
import csv
from flask.ext.script import Manager, Server, Command, Option
from ga_yelp_assignment import create_app
from ga_yelp_assignment.classifier_commands import (
    LoadTrainSaveClassifier,
    LoadTestClassifier,
    CalculateLabelMetrics)
from ga_yelp_assignment.commands import (
    LabelCounts,
    PhotoSizes)

application = create_app("Yelp Classifier")
manager = Manager(application)
port = int(os.environ.get('PORT', 5000))


def _make_context(app):
    return dict(app=app)


manager.add_command("runserver", Server(host='0.0.0.0', port=port))
manager.add_command("train_classifier", LoadTrainSaveClassifier())
manager.add_command("test_classifier", LoadTestClassifier())
manager.add_command("labels_counts", LabelCounts())
manager.add_command('photo_size_counts', PhotoSizes())
manager.add_command('calculate_metrics', CalculateLabelMetrics())


@manager.command
def list_routes():
    """List URLs of all application routes."""
    for rule in sorted(application.url_map.iter_rules(), key=lambda r: r.rule):
        print("{:10} {}".format(", ".join(rule.methods - set(['OPTIONS', 'HEAD'])), rule.rule))


if __name__ == '__main__':
    manager.run()
