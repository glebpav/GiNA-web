from flask import Flask, jsonify
from flask import request
from flask_cors import CORS

from scripts.circular_graph_builder import build_circular_graph
from scripts.script import run_script
from scripts.script_v3 import run_script_v3
from scripts.script_with_markers import run_script_with_markers

app = Flask(__name__)
CORS(app)


@app.route('/run-script', methods=['POST'])
def run_script():
    # print("get_json: ", request.get_json().get('sequence'))
    sequence = request.get_json().get('sequence')
    response = jsonify(run_script(sequence=sequence))
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/run-script-with-markers', methods=['POST'])
def run_script_2():
    # print("get_json: ", request.get_json().get('sequence'))
    structure = request.get_json().get('structure')
    response = jsonify(run_script_with_markers(structure_str=structure))
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/run-script3', methods=['POST'])
def run_script_3():
    # print("get_json: ", request.get_json().get('sequence'))
    structure = request.get_json().get('structure')
    response = jsonify(run_script_v3(structure=structure))
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
