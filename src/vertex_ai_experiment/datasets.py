import logging
import google.auth.credentials as auth_credentials
from google.cloud import aiplatform
from typing import Optional, List, Union, Sequence

logger = logging.getLogger(__name__)


def create_tabular_dataset(display_name: Optional[str] = None,
                           gcs_source: Optional[Union[str, Sequence[str]]] = None,
                           project: Optional[str] = None,
                           location: Optional[str] = None,
                           credentials: Optional[auth_credentials.Credentials] = None):
    # gcs_source = 'projects/my-project/location/us-central1/datasets/{DATASET_ID}'
    dataset = aiplatform.TabularDataset.create(gcs_source=gcs_source,
                                               display_name=display_name,
                                               project=project,
                                               location=location,
                                               credentials=credentials)
    return dataset


def retrieve_existing_tabular_dataset(dataset_name: str,
                                      project: Optional[str] = None,
                                      location: Optional[str] = None,
                                      credentials: Optional[auth_credentials.Credentials] = None):

    # dataset_name = 'projects/{my-project-number}/location/{my-location}/datasets/{DATASET_ID}'
    tabular_ds_dict = get_tabular_dataset_ids(project=project, location=location, credentials=credentials)
    dataset = aiplatform.TabularDataset(dataset_name=tabular_ds_dict.get(dataset_name),
                                        project=project,
                                        location=location,
                                        credentials=credentials)
    logger.info(f"dataset {dataset.display_name} retrieved")
    return dataset


def check_tabular_dataset_existence(dataset_name: str,
                                    project: Optional[str] = None,
                                    location: Optional[str] = None,
                                    credentials: Optional[auth_credentials.Credentials] = None):
    tabular_ds_dict = get_tabular_dataset_ids(project=project, location=location, credentials=credentials)
    try:
        _ = aiplatform.TabularDataset(dataset_name=tabular_ds_dict.get(dataset_name),
                                      project=project,
                                      location=location,
                                      credentials=credentials)
    except Exception as e:
        logger.warning(e)
        return False

    return True


def get_tabular_dataset_ids(project: Optional[str] = None,
                            location: Optional[str] = None,
                            credentials: Optional[auth_credentials.Credentials] = None):
    ds_list = aiplatform.TabularDataset.list(project=project, location=location, credentials=credentials)
    ds_dict = dict()
    for ds in ds_list:
        if ds.display_name not in ds_dict.keys():
            ds_dict[ds.display_name] = ds.name

    return ds_dict
