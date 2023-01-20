import google.auth.credentials as auth_credentials
import logging
import os
from google.cloud import aiplatform
from typing import Optional, List, Union

LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=LOGLEVEL)
logger = logging.getLogger()


def init_pipeline(project: Optional[str] = None,
                  location: Optional[str] = None,
                  experiment: Optional[str] = None,
                  staging_bucket: Optional[str] = None,
                  credentials: Optional[auth_credentials.Credentials] = None,
                  encryption_spec_key_name: Optional[str] = None
                  ):
    aiplatform.init(project=project,
                    location=location,
                    staging_bucket=staging_bucket,
                    experiment=experiment,
                    credentials=credentials,
                    encryption_spec_key_name=encryption_spec_key_name)


def get_custom_job_model(dataset: aiplatform.TabularDataset,
                         project: str,
                         location: str,
                         bucket: str,
                         display_name: str,
                         script_path: str,
                         container_uri: str,
                         model_serving_container_image_uri: str,
                         requirements: list,
                         replica_count: int = 1,
                         machine_type: str = "n1-standard-4",
                         accelerator_count: int = 1,
                         script_args:  Optional[List[Union[str, float, int]]] = None):
    # https://cloud.google.com/vertex-ai/docs/training/create-custom-job#create_custom_job-python
    # https://cloud.google.com/vertex-ai/docs/training/pre-built-containers
    logger.info("start ")
    init_pipeline(project=project,
                  location=location,
                  staging_bucket=bucket)
    logger.info("go job")
    job = aiplatform.CustomTrainingJob(
        display_name=display_name,
        script_path=script_path,
        container_uri=container_uri,
        requirements=requirements,
        model_serving_container_image_uri=model_serving_container_image_uri,
    )

    logger.info("run job")
    model = job.run(
        dataset=dataset,
        args=script_args,
        replica_count=replica_count,
        model_display_name=display_name,
        machine_type=machine_type,
        accelerator_count=accelerator_count
    )
    logger.info(f"display name : {model.display_name}")
    logger.info(f"ressource name : {model.resource_name}")
    logger.info(f"uri : {model.uri}")
    return model


def create_training_pipeline_custom_package_job_sample(
        project: str,
        location: str,
        staging_bucket: str,
        display_name: str,
        python_package_gcs_uri: str,
        python_module_name: str,
        container_uri: str,
        model_serving_container_image_uri: str,
        dataset_id: Optional[str] = None,
        model_display_name: Optional[str] = None,
        args: Optional[List[Union[str, float, int]]] = None,
        replica_count: int = 1,
        machine_type: str = "n1-standard-4",
        accelerator_type: str = "ACCELERATOR_TYPE_UNSPECIFIED",
        accelerator_count: int = 0,
        training_fraction_split: float = 0.8,
        validation_fraction_split: float = 0.1,
        test_fraction_split: float = 0.1,
        sync: bool = True):
    # https://cloud.google.com/vertex-ai/docs/training/create-training-pipeline
    # https://cloud.google.com/vertex-ai/docs/training/create-training-pipeline#aiplatform_create_training_pipeline_custom_job_sample-python

    init_pipeline(project=project,
                  location=location,
                  staging_bucket=staging_bucket)

    job = aiplatform.CustomPythonPackageTrainingJob(
        display_name=display_name,
        python_package_gcs_uri=python_package_gcs_uri,
        python_module_name=python_module_name,
        container_uri=container_uri,
        model_serving_container_image_uri=model_serving_container_image_uri,
    )

    dataset = aiplatform.ImageDataset(dataset_id) if dataset_id else None

    model = job.run(
        dataset=dataset,
        model_display_name=model_display_name,
        args=args,
        replica_count=replica_count,
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
        training_fraction_split=training_fraction_split,
        validation_fraction_split=validation_fraction_split,
        test_fraction_split=test_fraction_split,
        sync=sync,
    )

    model.wait()

    print(model.display_name)
    print(model.resource_name)
    print(model.uri)
    return model


def create_training_pipeline_custom_container_job_sample(
    project: str,
    location: str,
    staging_bucket: str,
    display_name: str,
    container_uri: str,
    model_serving_container_image_uri: str,
    dataset_id: Optional[str] = None,
    model_display_name: Optional[str] = None,
    args: Optional[List[Union[str, float, int]]] = None,
    replica_count: int = 1,
    machine_type: str = "n1-standard-4",
    accelerator_type: str = "ACCELERATOR_TYPE_UNSPECIFIED",
    accelerator_count: int = 0,
    training_fraction_split: float = 0.8,
    validation_fraction_split: float = 0.1,
    test_fraction_split: float = 0.1,
    sync: bool = True,
):
    # https://cloud.google.com/vertex-ai/docs/training/create-training-pipeline#custom-container
    
    # aiplatform.init(project=project, location=location, staging_bucket=staging_bucket)
    init_pipeline(project=project,
                  location=location,
                  staging_bucket=staging_bucket)

    job = aiplatform.CustomContainerTrainingJob(
        display_name=display_name,
        container_uri=container_uri,
        model_serving_container_image_uri=model_serving_container_image_uri,
    )

    # This example uses an ImageDataset, but you can use another type
    dataset = aiplatform.ImageDataset(dataset_id) if dataset_id else None

    model = job.run(
        dataset=dataset,
        model_display_name=model_display_name,
        args=args,
        replica_count=replica_count,
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
        training_fraction_split=training_fraction_split,
        validation_fraction_split=validation_fraction_split,
        test_fraction_split=test_fraction_split,
        sync=sync,
    )

    model.wait()

    print(model.display_name)
    print(model.resource_name)
    print(model.uri)
    return model
