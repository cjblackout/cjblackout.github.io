# Use the official Python image as the base image
FROM python:3.9

# Install system dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Clone the pymeanshift repository
RUN git clone https://github.com/fjean/pymeanshift.git

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Install the required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the working directory to pymeanshift
WORKDIR /pymeanshift

# Run the setup.py file
RUN python setup.py install

# Set the working directory in the container
WORKDIR /app

# Copy the application code to the container
COPY . .

# Expose the port that the application will run on
EXPOSE 8000

# Run the command to start the application
RUN pip install opencv-python

# Run the command to start the application
CMD ["uvicorn", "baseline:app", "--host", "0.0.0.0", "--port", "8000"]
