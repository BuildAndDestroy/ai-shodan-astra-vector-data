FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN chmod +x build_and_install.sh
RUN ./build_and_install.sh
RUN pip install -e .