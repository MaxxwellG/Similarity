# Docker file

FROM python:3.9

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "semantic.py"]

# for more informations visit https://docs.docker.com/develop/develop-images/dockerfile_best-practices/