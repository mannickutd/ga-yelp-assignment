
from flask import Flask


def create_app(app_name=None):
    application = Flask(__name__)

    application.config.from_object('config.Config')

    if app_name:
        application.config['APP_NAME'] = app_name

    from ga_yelp_assignment.views import main_bp

    application.register_blueprint(main_bp)

    return application
