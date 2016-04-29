from flask import Blueprint

main_bp = Blueprint('main', __name__,)

from ga_yelp_assignment.views import (
    main 
)
