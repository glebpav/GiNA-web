from flask import Flask, jsonify
from flask import request
from flask_cors import CORS

from scripts.circular_graph_builder import build_circular_graph
from scripts.graph_3d_builder import build_3d_graph

app = Flask(__name__)
CORS(app)


@app.route('/run-script3', methods=['POST'])
def run_script_3():
    structure = request.get_json().get('structure')
    response = jsonify(build_3d_graph(structure=structure))
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/circular-coords', methods=['POST'])
def circular_coords():
    structure = request.get_json().get('structure')
    response = jsonify(build_circular_graph(structure=structure))
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
