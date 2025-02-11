import warnings
from datetime import datetime
from crew import Icsagents

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run(input_data):
    """
    Run the crew with provided input data.
    """
    try:
        # Extract topic from input data
        topic = input_data.get('topic', 'MATHS')  # Default topic if not provided
        current_year = str(datetime.now().year)

        inputs = {'topic': topic, 'current_year': current_year}

        # Execute the multi-agent crew
        result = Icsagents().crew().kickoff(inputs=inputs)

        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}
