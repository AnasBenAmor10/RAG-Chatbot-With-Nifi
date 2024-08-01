# NiFi Environment

## Prerequisites

- Docker Desktop: Ensure Docker Desktop is installed and running on your system.

## Setup and Running Instructions

1. Build the Docker Containers:
   - For Windows: `docker-compose build`
   - For Linux/MacOS: `docker compose build`
2. Start the Docker Containers:
   - For Windows: `docker-compose up -d`
   - For Linux/MacOS: `docker compose up -d`

## Access to the User Interface

- Username: orthlane
- Password : orthlane

## Adding new processors

- Add the new processors' folder to the folder `nifi-python-extensions extracted_extension/extension`
- Add to the Dockerfile this line: `chmod +x /opt/nifi/nifi-current/python_extensions/name_folder_processeur`
