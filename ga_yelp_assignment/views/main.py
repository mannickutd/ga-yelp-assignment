
from ga_yelp_assignment.views import main_bp

from flask import (
    render_template,
    redirect,
    url_for,
    session,
    jsonify
)


@main_bp.route('/')
def home():
    return render_template('home.html')

