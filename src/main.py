# General packages
import argparse
import logging
import os
import sys
import yaml
import json
import warnings
import time
# import vertex_ai_experiment.training_job as vertex_jobs
import vertex_ai_experiment.models as vertex_model
import vertex_ai_experiment.datasets as vertex_dataset
import vertex_ai_experiment.containers as containers
from datetime import datetime
# from functools import reduce
from copy import deepcopy

# Google related packages
import google.auth as ga
from google.oauth2 import service_account

# Project packages
from cpr_dir.gcp_interface.bigquery_interface import BigqueryInterface
from cpr_dir.gcp_interface.storage_interface import StorageInterface

# from models.classifier import train_model_in_local, predict_in_local
from cpr_dir.models.vertex_classifier_predictor import get_predict_from_endpoint
from vertex_ai_experiment.models import get_endpoint_ids

# from vertex_ai_experiment.predictor import CprPredictor

warnings.filterwarnings("ignore", """Your application has authenticated using
end user credentials""")

NUMBER_ITERATION = 10
PARSER_TASK_CHOICE = ['train', 'predict']

DATA_CONFIG = {'features': {'fixed': ['Pclass', 'Age', 'SibSp', 'Parch'],
                            'age_col': 'Age',
                            'gender_col': 'Sex'
                            },
               'target_col': 'Survived',
               'passenger_id': 'PassengerId'
               }

if __name__ == '__main__':
    start = datetime.now()  # start of the script
    LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        level=LOGLEVEL)
    logger = logging.getLogger()
    # parse all given arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True, type=str,
                        choices=PARSER_TASK_CHOICE)
    parser.add_argument('--configuration', required=True, type=str)
    parser.add_argument('--env', required=False, type=str, choices=['local', 'cloud'], default='local')
    parser.add_argument('--endpoint_version', required=False, type=str, default='1')
    parser.add_argument('--model_suffix', required=False,
                        help="Has the format YYYYMMDD",
                        type=str,
                        default='20230116')

    args = parser.parse_args()

    # log input arguments:
    logger.info("Input arguments are :")
    for t, inpt in args.__dict__.items():
        logger.info(f"{t}: {inpt}")

    # retrieve infrastructure data and functional parameters
    with open(args.configuration, 'r') as f:
        config = yaml.safe_load(f)

    # global_cfg = get_global_config(project_account=project_account,
    #                                test_mode=args.test_mode == '1')

    # retrieve credentials
    if config['google_cloud']['credentials_json_file'] != "":
        credentials = service_account.Credentials.from_service_account_file(
            config['google_cloud']['credentials_json_file'])
    else:
        credentials, _ = ga.default()

    # instantiate a BQ interface
    bq_interface = BigqueryInterface(
        project_name=config['project_name'],
        dataset_name=config['dataset_name'],
        credentials=credentials)

    # instantiate a GS interface
    gs_interface = StorageInterface(
        project_name=config['project_name'],
        credentials=credentials)

    ml_units_used = 0
    # build the model name according to your need (e.g. by taking account of the current date or other information)
    model_name = 'titanic_rf_classifier'
    data_name = 'titanic_train' if args.task == 'train' else 'titanic_test'
    if args.env == 'cloud':  # args.task == 'train':
        logger.info('Running an ML-job')
        gcp_product = "Vertex AI"
        call_dir_path = os.path.dirname(os.path.abspath(__file__))
        project_dir_path = os.path.dirname(call_dir_path)
        # gather parameter for the job to run
        bucket_name = config['google_gcs'].get('bucket_name')
        titanic_dir = config['google_gcs'].get('directory_name')
        project_name = config['project_name']
        vertex_ai_location = config["vertex_ai"].get("location")

        model_serving_container_image_uri = config["vertex_ai"].get("serving_pre_built_container")
        container_uri = config["vertex_ai"].get("training_pre_built_container")

        # Setting up gcp parameters
        # stringify the configuration dictionary
        config_arg = json.dumps(config)
        data_arg = json.dumps(DATA_CONFIG)
        model_display_name_prefix = "cpr_titanic_survival_model"
        model_artifact_uri = f"gs://{bucket_name}/{titanic_dir}/cpr_model_artifact"
        if args.task == 'train':
            logger.info(" TRAINING ")
            # arguments_part = [['--project_name', project_name],
            #                  ['--configuration', config_arg],
            #                  ['--data_name', data_name],
            #                  ['--task', args.task],
            #                  ['--data_configuration', data_arg],
            #                  ['--model_name', model_name],
            #                  ['--model_artifact_gs_dir', model_artifact_uri]
            #                  ]

            # arguments = reduce(lambda v, w: v + w, arguments_part)

            logger.info(" Start custom training job with pre-built images")
            timestamp = time.strftime("%Y%m%d")  # time.strftime("%Y%m%d_%H%M%S")
            model_display_name = f'{model_display_name_prefix}_{timestamp}'

            path = os.path.abspath(__file__)
            dir_name = os.path.dirname(path)

            logger.info("A. Vertex Dataset stage")
            train_data_gcs = f"gs://{bucket_name}/{titanic_dir}/titanic_train.csv"
            if vertex_dataset.check_tabular_dataset_existence(dataset_name="titanic_train",
                                                              project=project_name,
                                                              location=vertex_ai_location,
                                                              credentials=credentials):
                logger.info("---- Dataset already available : fetching it")
                dataset = vertex_dataset.retrieve_existing_tabular_dataset(dataset_name="titanic_train",
                                                                           project=project_name,
                                                                           location=vertex_ai_location,
                                                                           credentials=credentials)
            else:
                logger.info("---- Dataset does not exist : creating it")
                dataset = vertex_dataset.create_tabular_dataset(display_name="titanic_train",
                                                                gcs_source=train_data_gcs,
                                                                project=project_name,
                                                                location=vertex_ai_location,
                                                                credentials=credentials)

            logger.info("B-1.: Create a model")
            logger.info("++++++ Skip and assume that a model has been trained and artifacts are available in Storage")
            # Here it is assumed that a model has already been trained and artifacts are available in Storage.

            logger.info("B-2.: Build Local Model")
            local_model = containers.create_local_model_container(
                path_to_source_dir=os.path.join(sys.path[0], 'cpr_dir'),
                location=vertex_ai_location,
                project_id=project_name,
                repository=config["vertex_ai"].get("repository"),
                image=config["vertex_ai"].get("image"),
                path_to_requirement=os.path.join(sys.path[0], 'cpr_dir', 'requirements.txt'),
                extra_packages=None)
            logger.info("B-2.: Push to Registry")
            logger.info(f"    The container spec is {local_model.get_serving_container_spec()}")
            containers.push_image_to_registry(local_model=local_model,
                                              location=vertex_ai_location,
                                              project_id=project_name)
            logger.info("C. Vertex Model stage")
            # need to point toward the built image
            model = vertex_model.upload_model(local_model=local_model,
                                              display_name=model_display_name,
                                              artifact_uri=model_artifact_uri,
                                              project=project_name,
                                              location=vertex_ai_location,
                                              credentials=credentials
                                              )
            logger.info("D. Creating Endpoint")
            # endpoint_display_name = f'endpoint_CPR_{model_display_name_prefix}_version_{model.version_id}'
            # endpoint = vertex_model.create_model_endpoint(display_name=endpoint_display_name,
            #                                              project=project_name,
            #                                              location=vertex_ai_location,
            #                                              credentials=credentials)
            logger.info("E. Deploying to endpoint")
            vertex_model.deploy_model(model=model,
                                      machine_type="n1-standard-2",
                                      deployed_model_display_name=model_display_name,
                                      endpoint=None)
                                      #endpoint)

        elif args.task == 'predict':
            logger.info("A. Retrieve test Data")

            prediction_data = gs_interface.storage_to_dataframe(bucket_name=bucket_name,
                                                                data_name="titanic_test.csv",
                                                                # data_name="titanic_test_vertex_ai_v1.csv",
                                                                gs_dir_path=titanic_dir)
            prediction_data_batch = deepcopy(prediction_data)

            logger.info("B-endpoint prediction: Retrieve Endpoint")
            endpoint_name = f'{model_display_name_prefix}_{args.model_suffix}_endpoint'
            endpoints = get_endpoint_ids(project=project_name, location=vertex_ai_location, credentials=credentials)
            endpoint = endpoints.get(endpoint_name)
            if endpoint is not None:
                project_id = endpoint.get('ressource_name').split('/')[1]
                # run endpoint prediction
                logger.info("C-endpoint prediction: Compute")
                # took less than 1 second for 87 predictions
                prediction_data['survival_prediction'] = get_predict_from_endpoint(
                    project=project_id,
                    endpoint_id=endpoint.get('name'),
                    location=vertex_ai_location,
                    instances=[list(r) for r in prediction_data.values]
                    )
                # instances={f'instance_key_{i+1}': r[i] for i in range(len(r))}

                logger.info("D-endpoint prediction: Push prediction to Storage")
                gs_interface.dataframe_to_storage(df=prediction_data,
                                                  bucket_name=bucket_name,
                                                  gs_dir_path=titanic_dir,
                                                  data_name="prediction_titanic_vertex_ai_v2_endpoint.csv")
            else:
                logger.info(f"endpoint {endpoint_name} not found")

            """
           # Only for informational purpose
           # run batch prediction
            logger.info("B-Batch prediction: Retrieve Model")
            model_name = f'{model_display_name_prefix}_{args.model_suffix}'
            models = get_model_ids(project=project_name, location=vertex_ai_location, credentials=credentials)
            model_infos = models.get(model_name)
            model = vertex_model.get_model(model_name=model_infos.get('name'),
                                           project=project_name,
                                           location=vertex_ai_location,
                                           credentials=credentials,
                                           version='1'
                                           )
 
            logger.info("C-Batch prediction: Compute")
            # took about 30 minutes for 87 predictions
            batch_predict_job = get_batch_prediction(
                model=model,
                gcs_source=[f'gs://{bucket_name}/{titanic_dir}/titanic_test_vertex_ai_v1.csv'],
                gcs_destination_prefix=f'gs://{bucket_name}//batch_prediction/results/',
                job_display_name=f'batch_prediction_{model_display_name_prefix}')
            # results are pushed directly ro Storage
            logger.info(f"Predictions are available in {batch_predict_job.output_info.gcs_output_directory}")
            # 'gs://dmp-y-tests_vertex_ai_bucket/batch_prediction/results/prediction-titanic_survival_model_20230116-2023_01_17T00_56_33_700Z'
            """
    else:
        logger.info("Run in local")
        # here depending on task
        if args.task == 'train':
            from cpr_dir.models.vertex_classifier import train_model_in_local
            # gs_dir_path = 'ai_platform_template_dir' = config.get('google_gcs').get('directory_name')
            bucket_name = config['google_gcs'].get('bucket_name')
            titanic_dir = config['google_gcs'].get('directory_name')
            project_name = config['project_name']
            vertex_ai_location = config["vertex_ai"].get("location")
            model_artifact_uri = f"gs://{bucket_name}/{titanic_dir}/cpr_model_artifact"
            train_data_df = gs_interface.storage_to_dataframe(bucket_name=bucket_name,
                                                              data_name="titanic_train.csv",
                                                              gs_dir_path=config['google_gcs'].get('directory_name'))
            train_data_df.dropna(inplace=True)

            train_model_in_local(
                train_data=train_data_df,
                gs_dir_path=config['google_gcs'].get('directory_name'),
                bucket_name=config['google_gcs'].get('bucket_name'),
                local_dir_path=config.get('local_dir_path', "tmp"),
                train_name=data_name,
                data_configuration=DATA_CONFIG,
                model_artifact_gs_dir=model_artifact_uri
            )

        else:
            pass

    logger.info(f"Task {args.task} is successfully completed.")
    # see https://cloud.google.com/vertex-ai/pricing#training for the pricing
