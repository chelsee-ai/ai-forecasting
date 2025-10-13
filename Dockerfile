# Use Python 3.12 slim image
FROM python:3.12-slim

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
WORKDIR /app
COPY . /app

# Cloud Run expects the PORT environment variable
ENV PORT 8080

# Expose the port
EXPOSE 8080

# Run your app
CMD ["python", "app.py"]
