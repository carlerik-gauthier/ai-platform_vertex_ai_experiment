from typing import List, Union

from google.cloud import aiplatform
from vertex_ai_experiment.models import make_batch_prediction_job

# run on endpoint : general overview
# https://github.com/googleapis/python-aiplatform/blob/main/samples/snippets/prediction_service/predict_custom_trained_model_sample.py
def get_predict_from_endpoint(
    project: str,
    endpoint_id: str,
    instances: Union[list, List[list]],
    location: str = "europe-west1"
):

    """
    `instances` can be either single instance of type dict or a list
    of instances.
    """
    api_endpoint: str = f"{location}-aiplatform.googleapis.com"
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    # The format of each instance should conform to the deployed model's prediction input schema.
    instances = instances if type(instances[0]) == list else [instances]

    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(endpoint=endpoint, instances=instances)
    prediction = response.predictions
    return prediction


def get_batch_prediction(
        model: aiplatform.Model,
        job_display_name: str,
        gcs_source: List[str],
        gcs_destination_prefix: str
) -> aiplatform.BatchPredictionJob:
    job = make_batch_prediction_job(model=model,
                                    job_display_name=job_display_name,
                                    gcs_source=gcs_source,
                                    gcs_destination_prefix=gcs_destination_prefix,
                                    machine_type='n1-standard-4',
                                    instances_format='csv')

    return job
