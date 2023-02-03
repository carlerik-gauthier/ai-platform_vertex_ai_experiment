"""
This script contains model to be trained
"""
import argparse
import os
import sys
import logging
import warnings
import json
import pickle
import joblib

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as mse

import google.auth as ga
from google.oauth2 import service_account
from google.cloud import storage

# custom modules
# from gcp_interface.storage_interface import StorageInterface
from cpr_dir.preprocessing.vertex_titanic_preprocessor import TitanicPreprocessor

warnings.filterwarnings("ignore", """Your application has authenticated using
end user credentials""")
LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=LOGLEVEL)
logger = logging.getLogger()

TARGET_COL = ''


def get_rf_classification_model(train_data: pd.DataFrame,
                                feature_cols: list,
                                target_col: str,
                                n_splits: int = 5,
                                n_estimators: int = 10,
                                seed: int = 123
                                ):
    # reset indices for safety purpose
    train_data.reset_index(drop=True, inplace=True)
    # Random Forest Classifier model init
    rf_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
    # k-fold init
    k_fold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    i = 1
    for train_index, test_index in k_fold.split(train_data):
        train_x = train_data.loc[train_index, feature_cols]
        train_y = train_data.loc[train_index, target_col]

        test_x = train_data.loc[test_index, feature_cols]
        test_y = train_data.loc[test_index, target_col]

        # Fitting the model
        rf_classifier.fit(train_x, train_y)

        # Predict the model
        pred = rf_classifier.predict(test_x)

        # RMSE Computation
        rmse = np.sqrt(mse(test_y, pred))
        logger.info(f"Fold {i}, RMSE : {rmse}")
        i += 1

    return rf_classifier


def train_model_in_local(train_data: pd.DataFrame,
                         bucket_name: str,
                         gs_dir_path: str,
                         local_dir_path: str,
                         train_name: str,
                         data_configuration: dict,
                         model_artifact_gs_dir: str = ''
                         ):

    logger.info(f" TEST model_artifact_gs_dir = {model_artifact_gs_dir}")
    model_directory = os.environ['AIP_MODEL_DIR'] if model_artifact_gs_dir == '' else model_artifact_gs_dir
    logger.info(f" TEST model_directory = {model_directory}")
    # preprocess
    # feat_dict = data_configuration.get('features')
    # target_col = data_configuration.get('target_col')
    # age_col = feat_dict.get('age_col')
    # gender_col = feat_dict.get('gender_col')
    preprocessor = TitanicPreprocessor()

    data_df = preprocessor.preprocess(df=train_data
                                      )
    target_col = data_configuration.get('target_col')
    feature_cols = [c for c in data_df.columns if c != target_col]
    logger.info(f" [MODEL TRAIN] Feature columns are : {feature_cols}. \n Target column is {target_col}")
    data_df[target_col] = train_data[target_col]
    model = get_rf_classification_model(train_data=data_df,
                                        feature_cols=feature_cols,
                                        target_col=target_col
                                        )

    # save model as a pickle file in local first
    # https://cloud.google.com/vertex-ai/docs/training/exporting-model-artifacts#scikit-learn
    model_artifact_filename = 'model.joblib'
    preprocessor_artifact_filename = 'preprocessor.pkl'
    # data_config_artifact_filename = 'data_config.json'

    # NB : le nom de l'artifact n'est pas optionnel : il doit s'appeler model.pkl ou model.joblib
    for obj, artifact in {"model": (model, model_artifact_filename),
                          "preprocessor": (preprocessor, preprocessor_artifact_filename),
                          # 'data_config': (data_configuration, data_config_artifact_filename)
                          }.items():
        artifact_obj, artifact_file = artifact
        file_local_path = os.path.join(local_dir_path, artifact_file)
        logger.info(f" -- Dumping {obj} to {file_local_path}")
        if artifact_file.split('.')[-1] == 'pkl':
            pickle.dump(artifact_obj, open(file_local_path, 'wb'))
        elif artifact_file.split('.')[-1] == 'joblib':
            joblib.dump(artifact_obj, open(file_local_path, 'wb'))
        else:
            json.dump(artifact_obj, open(file_local_path, 'w'))
        # Upload output to Storage
        # model_directory = os.environ['AIP_MODEL_DIR']
        storage_path = os.path.join(model_directory, artifact_file)
        blob = storage.blob.Blob.from_string(storage_path, client=storage.Client())
        blob.upload_from_filename(file_local_path)

    # return model


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


if __name__ == '__main__':
    # parse all given arguments
    # https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/sdk/SDK_End_to_End_Tabular_Custom_Training.ipynb

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project_name', required=True, type=str)
    parser.add_argument('--configuration', required=True, type=str)
    parser.add_argument('--data_name', required=True, type=str)
    parser.add_argument('--task', required=True, type=str)
    parser.add_argument('--data_configuration', required=True, type=str)
    parser.add_argument('--model_name', required=True, type=str)
    parser.add_argument('--model_artifact_gs_dir', required=False, default='')

    args = parser.parse_args()

    logger.info(f'READY to launch {args.task} task in the cloud')
    # inverse operation : turns a string-ified dictionary to an actual dictionary
    infra_config = json.loads(args.configuration)
    data_config = json.loads(args.data_configuration)
    # retrieve credentials
    if infra_config['google_cloud']['credentials_json_file'] != "":
        credentials = service_account.Credentials.from_service_account_file(
            infra_config['google_cloud']['credentials_json_file'])
    else:
        credentials, _ = ga.default()

    train_data_df = aip_data_to_dataframe(wild_card_path=os.environ["AIP_TRAINING_DATA_URI"],
                                          project_name=args.project_name)

    train_model_in_local(
        train_data=train_data_df,
        gs_dir_path=infra_config['google_gcs'].get('directory_name'),
        bucket_name=infra_config['google_gcs'].get('bucket_name'),
        local_dir_path=infra_config.get('local_dir_path', "tmp"),
        train_name=args.data_name,
        data_configuration=data_config,
        model_artifact_gs_dir=args.model_artifact_gs_dir
    )
