# Use official Python 3.10 slim image
FROM python:3.10-slim

# Avoid prompts during install
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies first
RUN apt-get update && \
    apt-get install -y ffmpeg libgl1 libglib2.0-0 build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first to use Docker cache
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Expose port for Flask
EXPOSE 5000

# Command to run Flask
CMD ["python", "app.py"]
