from google.cloud.aiplatform.prediction import LocalModel
from google.cloud import aiplatform
from cpr_dir.predictor import CprPredictor
# from cpr_dir.handler import CprHandler
from typing import Optional, List


def create_local_model_container(path_to_source_dir: str,
                                 location: str,
                                 project_id: str,
                                 repository: str,
                                 image: str,
                                 path_to_requirement: str,
                                 extra_packages: Optional[List[str]] = None):
    """
    https://cloud.google.com/python/docs/reference/aiplatform/1.19.1/google.cloud.aiplatform.prediction.LocalModel#google_cloud_aiplatform_prediction_LocalModel_build_cpr_model
    :param path_to_source_dir:
    :param location:
    :param project_id: Of the form "123456789"
    :param repository:
    :param image:
    :param path_to_requirement:
    :param extra_packages
    :return:
    """

    aiplatform.init(project=project_id, location=location)

    local_model = LocalModel.build_cpr_model(
        src_dir=path_to_source_dir,
        output_image_uri=f"{location}-docker.pkg.dev/{project_id}/{repository}/{image}",
        predictor=CprPredictor,
        # handler=CprHandler,
        requirements_path=path_to_requirement,
        extra_packages=extra_packages)

    return local_model


def push_image_to_registry(local_model: LocalModel,
                           location: str,
                           project_id: str,
                           ):
    aiplatform.init(project=project_id, location=location)
    local_model.push_image()
