from flask import Flask, request, jsonify
import logging
from main import run

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.route('/run', methods=['POST'])
def run_multiagent():
    try:
        data = request.json
        logging.info(f"Received request data: {data}")
        
        run()  # Call the multi-agent run function
        
        return jsonify({"message": "Multi-agent system executed successfully"}), 200
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)