# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# git is often required for installing python packages from git repositories
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Note: By default, this installs the CUDA version of PyTorch which is large.
# If you are deploying to a CPU-only environment (like standard Cloud Run),
# you can reduce image size by installing the CPU version explicitly:
# RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
#     pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port 8080 (Cloud Run default)
EXPOSE 8080

# Define environment variables
# ensuring python output is sent directly to terminal without buffering
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Run server.py when the container launches
CMD ["python", "server.py"]
