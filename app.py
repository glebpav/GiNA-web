import json

from flask import Flask
from flask import request, Blueprint
from flask_cors import CORS

from api_config import CIRCULAR_GRAPH_URN, GRAPH_3D_URN
from scripts.circular_graph_builder import build_circular_graph
from scripts.graph_3d_builder import build_3d_graph



g2d_bp = Blueprint('g2d', __name__, url_prefix=f'/{CIRCULAR_GRAPH_URN}')
g3d_bp = Blueprint('g3d', __name__, url_prefix=f'/{GRAPH_3D_URN}')


def create_app():
    app = Flask(__name__)
    CORS(app)
    
    app.register_blueprint(g2d_bp)
    app.register_blueprint(g3d_bp)

    return app


@g2d_bp.post('/')
def circular_graph():
    data = request.get_json()
    if isinstance(data, str):
        data = json.loads(data)
        
    structure = data.get('structure')
    response = build_circular_graph(structure=structure)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@g3d_bp.post('/')
def graph_3d():
    data = request.get_json()
    if isinstance(data, str):
        data = json.loads(data)
        
    structure = data.get('structure')
    response = build_3d_graph(structure=structure)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response




