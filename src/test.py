import os
import sys
import logging
import warnings
import json
import pickle

from typing import Union
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as mse
from typing import Optional
from copy import deepcopy

import google.auth as ga
import google.auth.credentials as auth_credentials
from google.oauth2 import service_account
from google.cloud import storage, aiplatform

from gcp_interface.storage_interface import StorageInterface

warnings.filterwarnings("ignore", """Your application has authenticated using
end user credentials""")
LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=LOGLEVEL)
logger = logging.getLogger()
DATA_CONFIG = {'features': {'fixed': ['Pclass', 'Age', 'SibSp', 'Parch'],
                            'age_col': 'Age',
                            'gender_col': 'Sex'
                            },
               'target_col': 'Survived',
               'passenger_id': 'PassengerId'
               }

def aip_data_to_dataframe(wild_card_path: str, project_name: str):
    # see https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/sdk/SDK_End_to_End_Tabular_Custom_Training.ipynb

    logger.info(f"[STORAGE] Looking at the following uris list :\n {wild_card_path}")

    core_wildcard = wild_card_path.replace("gs://", '').replace('*', '')
    core_wildcard_parts = core_wildcard.split('/')
    bucket_name = core_wildcard_parts[0]
    prefix = '/'.join(core_wildcard_parts[1:])
    gs_client = storage.Client(project=project_name)
    bucket = gs_client.bucket(bucket_name=bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    dfs = map(lambda blob: pd.read_csv(f"gs://{bucket_name}/"+blob.name), blobs)
    try:
        df = pd.concat(dfs, ignore_index=True).drop(columns='Unnamed: 0')
    except KeyError:
        blobs = bucket.list_blobs(prefix=prefix)
        dfs = map(lambda blob: pd.read_csv(f"gs://{bucket_name}/"+blob.name), blobs)
        df = pd.concat(dfs, ignore_index=True)
    except ValueError:
        logger.error("Data is NOT available in Storage")
        sys.exit(1)

    return df


def get_model_ids(project: Optional[str] = None,
                  location: Optional[str] = None,
                  credentials: Optional[auth_credentials.Credentials] = None):
    model_list = aiplatform.Model.list(project=project, location=location, credentials=credentials)
    model_dict = dict()
    for model in model_list:
        if model.display_name not in model_dict.keys():
            model_dict[model.display_name] = model
            # {'name': model.name, 'ressource_name': model.resource_name, 'version_id': model.version_id}

    return model_dict


def get_endpoint_ids(project: Optional[str] = None,
                  location: Optional[str] = None,
                  credentials: Optional[auth_credentials.Credentials] = None):
    endpoint_list = aiplatform.Endpoint.list(project=project, location=location, credentials=credentials)
    endpoint_dict = dict()
    for endpoint in endpoint_list:
        if endpoint.display_name not in endpoint_dict.keys():
            endpoint_dict[endpoint.display_name] = endpoint # {'name': endpoint.name, 'ressource_name': endpoint.resource_name}

    return endpoint_dict

#aip_data_to_dataframe(
#    wild_card_path="gs://dmp-y-tests_vertex_ai_bucket/ai_platform_template_dir/vertex_exp/aiplatform-custom-training-2023-01-12-18:01:03.644/dataset-4013797983521865728-tables-2023-01-12T17:01:04.062449Z/training-*",
#     project_name='dmp-y-tests')

"""
print(get_model_ids(project="dmp-y-tests",
              location="europe-west1")
      )
"""
#models = get_model_ids(project="dmp-y-tests",
#              location="europe-west1")

#endpoints = get_endpoint_ids(project="dmp-y-tests",
#              location="europe-west1")

def preprocess_titanic_test_file(application_data: pd.DataFrame,
                                 data_configuration: dict
                                 ) -> pd.DataFrame:
    from preprocessing.titanic_preprocess import preprocess

    # preprocess
    feat_dict = data_configuration.get('features')
    age_col = feat_dict.get('age_col')
    gender_col = feat_dict.get('gender_col')
    data_df = preprocess(df=application_data,
                         age_col=age_col,
                         gender_col=gender_col,
                         fixed_columns=feat_dict.get('fixed', [age_col, gender_col]),
                         )
    target_col = data_configuration.get('target_col')
    feature_cols = [c for c in data_df.columns if c != target_col]
    return deepcopy(data_df[feature_cols])


def get_data_from_storage(gs_interface: StorageInterface,
                          data_name: str,
                          bucket_name: str,
                          gs_dir_path: str = None):
    data = gs_interface.storage_to_dataframe(bucket_name=bucket_name,
                                             data_name=data_name,
                                             gs_dir_path=gs_dir_path)
    data.dropna(inplace=True)
    # data['product_code'] = data['product_code'].astype('int32').astype('str')
    return data


storage_interface = StorageInterface(project_name='dmp-y-tests')

# test_data = get_data_from_storage(gs_interface=storage_interface,
#                                  bucket_name='dmp-y-tests_vertex_ai_bucket',
#                                  gs_dir_path='ai_platform_template_dir',
#                                  data_name='titanic_test_vertex_ai_v1.csv')

"""
test_data_proc = preprocess_titanic_test_file(application_data=test_data,
                                              data_configuration=DATA_CONFIG)

storage_interface.dataframe_to_storage(bucket_name='dmp-y-tests_vertex_ai_bucket',
                                       gs_dir_path='ai_platform_template_dir',
                                       data_name='titanic_test_vertex_ai_v1.csv',
                                       df=test_data_proc)
"""

#wild_card_path = "gs://dmp-y-tests_vertex_ai_bucket/custom_python_package/vertex_exp/aiplatform-custom-training-2023-01-18-16:47:38.138/dataset-4013797983521865728-tables-2023-01-18T15:47:38.792585Z/training-*"
#core_wildcard = wild_card_path.replace("gs://", '').replace('*', '')
#core_wildcard_parts = core_wildcard.split('/')
#bucket_name = core_wildcard_parts[0]
#dir_path = '/'.join(core_wildcard_parts[1:-1])

#df = storage_interface.storage_to_dataframe(bucket_name=bucket_name, gs_dir_path=dir_path, data_name=core_wildcard_parts[-1])
#df.head()

model_dir="gs://dmp-y-tests_vertex_ai_bucket/custom_python_package/vertex_exp/jobs/aiplatform-custom-training-2023-01-19-10:19:25.524/model"
storage_dir_path = '/'.join(model_dir.replace("gs://", '').split('/')[1:])
print(storage_dir_path)