# Dockerfile extension of Ultralytics' Docker image by JavierMtz5

FROM ultralytics/ultralytics:latest

# Copy the requirements file and install dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the model and the training/test data
COPY ./app ${APP_HOME}/code/app

CMD ["fastapi", "run", "code/app/main.py", "--port", "8080"]