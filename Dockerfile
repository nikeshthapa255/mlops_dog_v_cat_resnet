# Use a Debian-based Python image
FROM python:3.10-slim

# Set environment variables with correct syntax
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies required by TensorFlow and other packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Specify working directory
WORKDIR /dog_vs_cat_api

# Copy application files
COPY /dog_vs_cat_api /dog_vs_cat_api/

# Update pip and install Python dependencies
RUN pip install --upgrade pip 
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for the application
EXPOSE 8001

# Start FastAPI application
CMD ["python", "app/main.py"]
