from flask import Flask, request, jsonify
import logging
import threading
import uuid
from main import run

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Dictionary to store job results
job_results = {}

def run_multiagent_async(job_id, input_data):
    """Runs the multi-agent process asynchronously and stores the result."""
    try:
        logging.info(f"Starting job {job_id} with data: {input_data}")

        # Call the multi-agent function with dynamic inputs
        result = run(input_data)
        
        job_results[job_id] = {"status": "completed", "result": result}
    except Exception as e:
        logging.error(f"Error in job {job_id}: {str(e)}")
        job_results[job_id] = {"status": "failed", "error": str(e)}

@app.route('/run', methods=['POST'])
def run_multiagent():
    try:
        data = request.json
        job_id = str(uuid.uuid4())  # Generate a unique job ID
        job_results[job_id] = {"status": "processing"}  # Mark as processing

        # Start the process in a separate thread
        thread = threading.Thread(target=run_multiagent_async, args=(job_id, data))
        thread.start()

        return jsonify({"message": "Multi-agent system started", "job_id": job_id}), 202
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/status/<job_id>', methods=['GET'])
def get_status(job_id):
    """Check the status of a job."""
    if job_id in job_results:
        return jsonify(job_results[job_id])
    return jsonify({"error": "Job ID not found"}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
