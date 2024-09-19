# Use an official Python runtime as a parent image
FROM python:3.10-slim-buster

# Set the working directory to /app
WORKDIR /app 
RUN mkdir -p /app/dataset /app/reference /app/logs

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

    # Install pipenv
RUN pip install --upgrade pip && pip install pipenv

# Set the working directory to /app
WORKDIR /app 

# Copy the current directory contents into the container at /app
#COPY . /app

# Copy the Pipfile and Pipfile.lock first, to ensure that Docker will cache this step if dependencies don’t change
COPY Pipfile Pipfile.lock /app/

# Install the dependencies
RUN pipenv install --deploy --system

COPY . /app

CMD ["python", "./client/client.py", "--datasetpath", "/app/dataset", "--referencepath", "/app/reference", "--dataset", "OCT"]
