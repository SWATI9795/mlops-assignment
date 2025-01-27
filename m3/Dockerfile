# Use official Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements and install them
COPY requirements.txt /app/
RUN pip install -r requirements.txt

# Copy the Flask app and model file
COPY app.py /app/
COPY model.pkl /app/

# Expose the port the app runs on
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]

