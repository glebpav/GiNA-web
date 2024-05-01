from flask import Flask
from flask import request
from flask_cors import CORS

from api_config import CIRCULAR_GRAPH_URN, GRAPH_3D_URN
from scripts.circular_graph_builder import build_circular_graph
from scripts.graph_3d_builder import build_3d_graph

app = Flask(__name__)
CORS(app)


@app.route(f'/{CIRCULAR_GRAPH_URN}', methods=['POST'])
def circular_graph():
    structure = request.get_json().get('structure')
    response = build_circular_graph(structure=structure)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route(f'/{GRAPH_3D_URN}', methods=['POST'])
def graph_3d():
    structure = request.get_json().get('structure')
    response = build_3d_graph(structure=structure)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
