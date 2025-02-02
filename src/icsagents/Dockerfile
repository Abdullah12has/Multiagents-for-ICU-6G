# Use an official Python base image
FROM python:3.10

# Set the working directory
WORKDIR /

# Copy the application files into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir crewai requests

# Set environment variable to point to Ollama service
ENV OLLAMA_API_BASE="http://ollama:11434"

# Run the application
CMD ["python", "main.py"]
