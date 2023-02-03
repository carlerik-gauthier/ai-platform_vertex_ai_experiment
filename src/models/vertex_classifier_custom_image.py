# https://cloud.google.com/python/docs/reference/aiplatform/latest#overview

import argparse
import os
# import sys
import logging
import warnings
import json
import pickle

# from typing import Union
# from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as mse

import google.auth as ga
from google.oauth2 import service_account
# from google.cloud import storage

# custom modules
from gcp_interface.storage_interface import StorageInterface
from preprocessing.titanic_preprocess import preprocess

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


def get_titanic_survival_prediction(model: RandomForestClassifier,
                                    application_data: pd.DataFrame,
                                    data_configuration: dict
                                    ) -> pd.DataFrame:
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

    # make prefiction
    logger.info(f" [MODEL PREDICT] Feature columns to be used are : {feature_cols}.")
    predictions = model.predict(X=data_df[feature_cols])
    proba_predictions = model.predict_proba(X=data_df[feature_cols])
    # turn predictions to a dataframe
    passenger_id = data_configuration.get('passenger_id')
    predictions_df = pd.DataFrame(data={passenger_id: application_data[passenger_id],
                                        'predicted_survival_probability': proba_predictions.T[1],
                                        'predicted_survival': predictions,
                                        }
                                  )
    return predictions_df


def train_model_in_local(train_data: pd.DataFrame,
                         gs_interface: StorageInterface,
                         bucket_name: str,
                         # gs_dir_path: str,
                         local_dir_path: str,
                         # train_name: str,
                         data_configuration: dict,
                         # model_name: str
                         ):
    # Retrieve data from Storage
    # train_data = get_data_from_storage(
    #    gs_interface=gs_interface,
    #    data_name=train_name + '.csv',
    #    bucket_name=bucket_name,
    #    gs_dir_path=gs_dir_path)

    # preprocess
    feat_dict = data_configuration.get('features')
    target_col = data_configuration.get('target_col')
    age_col = feat_dict.get('age_col')
    gender_col = feat_dict.get('gender_col')
    data_df = preprocess(df=train_data,
                         age_col=age_col,
                         gender_col=gender_col,
                         fixed_columns=[target_col]+feat_dict.get('fixed', [age_col, gender_col]),
                         )
    target_col = data_configuration.get('target_col')
    feature_cols = [c for c in data_df.columns if c != target_col]
    logger.info(f" [MODEL TRAIN] Feature columns are : {feature_cols}. \n Target column is {target_col}")
    model = get_rf_classification_model(train_data=data_df,
                                        feature_cols=feature_cols,
                                        target_col=target_col
                                        )
    # save model as a pickle file in local first
    # https://cloud.google.com/vertex-ai/docs/training/exporting-model-artifacts#scikit-learn
    artifact_filename = 'model.pkl'
    # NB : le nom de l'artifact n'est pas optionnel : il doit s'appeler model.pkl ou model.joblib

    file_local_path = os.path.join(local_dir_path, artifact_filename)
    logger.info(f" -- Dumping model to {file_local_path}")
    pickle.dump(model, open(file_local_path, 'wb'))
    # Upload output to Storage
    model_directory = os.environ['AIP_MODEL_DIR']
    storage_path = os.path.join(model_directory, artifact_filename)
    logger.info(f"""Uploading model to Storage : {storage_path}"""
                )

    # remove gs://{bucket_name} from model_directory
    storage_dir_path = '/'.join(model_directory.replace("gs://", '').split('/')[1:])
    logger.info(f"Local directory is {local_dir_path}")
    gs_interface.local_to_storage(data_name=artifact_filename,
                                  bucket_name=bucket_name,
                                  local_dir_path=local_dir_path,
                                  storage_dir_path=storage_dir_path
                                  )


"""
def aip_data_to_dataframe(wild_card_path: str, project_name: str):
    # see https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/sdk/
    SDK_End_to_End_Tabular_Custom_Training.ipynb

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
"""


def aip_data_to_dataframe(wild_card_path: str, gs_interface: StorageInterface):
    # see https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/
    # community/sdk/SDK_End_to_End_Tabular_Custom_Training.ipynb

    logger.info(f"[STORAGE] Looking at the following wild card path :\n {wild_card_path}")

    core_wildcard = wild_card_path.replace("gs://", '').replace('*', '')
    core_wildcard_parts = core_wildcard.split('/')

    df = gs_interface.storage_to_dataframe(bucket_name=core_wildcard_parts[0],
                                           gs_dir_path='/'.join(core_wildcard_parts[1:-1]),
                                           data_name=core_wildcard_parts[-1])

    return df


if __name__ == '__main__':
    # parse all given arguments
    # https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/sdk/SDK_End_to_End_Tabular_Custom_Training.ipynb
    # https://cloud.google.com/vertex-ai/docs/training/code-requirements?hl=fr
    # https://cloud.google.com/vertex-ai/docs/reference/rest/v1/CustomJobSpec#FIELDS.base_output_directory
    # https://cloud.google.com/vertex-ai/docs/training/using-managed-datasets?hl=fr

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project_name', required=True, type=str)
    parser.add_argument('--configuration', required=True, type=str)
    parser.add_argument('--data_name', required=True, type=str)
    parser.add_argument('--task', required=True, type=str)
    parser.add_argument('--data_configuration', required=True, type=str)
    parser.add_argument('--model_name', required=True, type=str)

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

    # instantiate a GS interface
    storage_interface = StorageInterface(
        project_name=args.project_name,
        credentials=credentials)

    train_data_df = aip_data_to_dataframe(wild_card_path=os.environ["AIP_TRAINING_DATA_URI"],
                                          gs_interface=storage_interface)

    # train_data_df = aip_data_to_dataframe(wild_card_path=os.environ["AIP_TRAINING_DATA_URI"],
    #                                       project_name=args.project_name)

    if args.task == 'train':
        train_model_in_local(
            train_data=train_data_df,
            # gs_dir_path=infra_config['google_gcs'].get('directory_name'),
            bucket_name=infra_config['google_gcs'].get('bucket_name'),
            local_dir_path=infra_config.get('local_dir_path', "tmp"),
            # train_name=args.data_name,
            data_configuration=data_config,
            gs_interface=storage_interface
            # model_name=args.model_name
        )
    elif args.task == 'predict':
        pass
