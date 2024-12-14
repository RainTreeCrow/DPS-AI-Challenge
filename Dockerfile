# Use an appropriate Python base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy all files to the container
COPY . .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port Cloud Run uses
EXPOSE 8080

# Command to run the Flask app
CMD ["python", "app.py"]