FROM gcr.io/deeplearning-platform-release/sklearn-cpu.0-23
# https://console.cloud.google.com/gcr/images/deeplearning-platform-release/GLOBAL

WORKDIR /src

COPY src /src

RUN pip install -r vertex_ai_reqs.txt

ENTRYPOINT ["python", "-m", "models.vertex_classifier_custom_image"]