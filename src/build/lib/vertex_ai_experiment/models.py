import google.auth.credentials as auth_credentials
from google.cloud import aiplatform
from typing import Optional, List
# import datasets as dt

# https://github.com/googleapis/python-aiplatform


def get_model(model_name: str,
              project: Optional[str] = None,
              location: Optional[str] = None,
              credentials: Optional[auth_credentials.Credentials] = None,
              version: Optional[str] = None
              ) -> aiplatform.Model:
    model = aiplatform.Model(model_name=model_name,
                             project=project,
                             location=location,
                             version=version,
                             credentials=credentials)
    return model


def upload_model(serving_container_image_uri: Optional[str] = None,
                 artifact_uri: Optional[str] = None,
                 display_name: Optional[str] = None,
                 project: Optional[str] = None,
                 location: Optional[str] = None,
                 credentials: Optional[auth_credentials.Credentials] = None):
    """
    Upload model from Model registry
    """
    model = aiplatform.Model.upload(
        serving_container_image_uri=serving_container_image_uri,
        artifact_uri=artifact_uri,
        display_name=display_name,
        project=project,
        location=location,
        credentials=credentials
    )

    return model


def make_batch_prediction_job(model: aiplatform.Model,
                              job_display_name: str,
                              gcs_source: List[str],
                              gcs_destination_prefix,
                              machine_type: str = 'n1-standard-4',
                              instances_format: str = 'csv') -> aiplatform.BatchPredictionJob:
    # gcs_source = ['gs://path/to/my/file.csv']
    # gcs_destination_prefix='gs://path/to/by/batch_prediction/results/'
    # https://cloud.google.com/python/docs/reference/aiplatform/latest/index.html#batch-prediction

    batch_prediction_job = model.batch_predict(
        job_display_name=job_display_name,
        instances_format=instances_format,
        machine_type=machine_type,
        gcs_source=gcs_source,
        gcs_destination_prefix=gcs_destination_prefix
        )

    return batch_prediction_job


def create_model_endpoint(display_name: Optional[str] = None,
                          project: Optional[str] = None,
                          location: Optional[str] = None,
                          credentials: Optional[auth_credentials.Credentials] = None,
                          endpoint_id: Optional[str] = None,
                          create_request_timeout: Optional[float] = 300) -> aiplatform.Endpoint:
    endpoint = aiplatform.Endpoint.create(display_name=display_name,
                                          project=project,
                                          location=location,
                                          credentials=credentials,
                                          create_request_timeout=create_request_timeout,
                                          endpoint_id=endpoint_id)
    return endpoint


def deploy_model(model: aiplatform.Model,
                 deployed_model_display_name: str,
                 endpoint: aiplatform.Endpoint) -> None:
    model.deploy(endpoint=endpoint, deployed_model_display_name=deployed_model_display_name, traffic_split={"0": 100})

    model.wait()


def get_model_ids(project: Optional[str] = None,
                  location: Optional[str] = None,
                  credentials: Optional[auth_credentials.Credentials] = None):
    model_list = aiplatform.Model.list(project=project, location=location, credentials=credentials)
    model_dict = dict()
    for model in model_list:
        if model.display_name not in model_dict.keys():
            model_dict[model.display_name] = {'name': model.name,
                                              'version_id': model.version_id,
                                              'ressource_name': model.resource_name}

    return model_dict


def get_endpoint_ids(project: Optional[str] = None,
                     location: Optional[str] = None,
                     credentials: Optional[auth_credentials.Credentials] = None):
    endpoint_list = aiplatform.Endpoint.list(project=project, location=location, credentials=credentials)
    endpoint_dict = dict()
    for endpoint in endpoint_list:
        if endpoint.display_name not in endpoint_dict.keys():
            endpoint_dict[endpoint.display_name] = {'name': endpoint.name,
                                                    'ressource_name': endpoint.resource_name}

    return endpoint_dict
