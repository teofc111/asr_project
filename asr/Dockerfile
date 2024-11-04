# Get python image
FROM python:3.12.7-slim

# Set working directory
WORKDIR /app

# Install system dependencies and then clean up
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements text to container and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy main app
COPY asr_api.py .

# Expose port 8001
EXPOSE 8001

# Run the Flask app
CMD ["python", "asr_api.py"]
