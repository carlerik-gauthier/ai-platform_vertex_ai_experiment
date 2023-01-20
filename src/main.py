# General packages
import argparse
import logging
import os
import sys
import yaml
import json
import warnings
import time
from datetime import datetime
from functools import reduce

# Google related packages
import google.auth as ga
from google.oauth2 import service_account
from googleapiclient import discovery

# Project packages
from gcp_interface.bigquery_interface import BigqueryInterface
from gcp_interface.storage_interface import StorageInterface

from gcp_interface.ml_job import get_job_id, get_job_body, is_success, \
    get_consumed_ml_units

from models.classifier import train_model_in_local, predict_in_local

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

    args = parser.parse_args()

    # log input arguments:
    logger.info("Input arguments are :")
    for t, inpt in args.__dict__.items():
        logger.info(f"{t}: {inpt}")

    # retrieve infrastructure data and functional parameters
    with open(args.configuration, 'r') as f:
        config = yaml.safe_load(f)

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
    if args.env == 'cloud':
        logger.info('Running an ML-job')
        gcp_product = "AI-Platform"
        call_dir_path = os.path.dirname(os.path.abspath(__file__))
        project_dir_path = os.path.dirname(call_dir_path)
        # gather parameter for the job to run
        bucket_name = config['google_gcs'].get('bucket_name')
        project_name = config['project_name']

        # Setting up gcp parameters
        # stringify the configuration dictionary
        config_arg = json.dumps(config)
        data_arg = json.dumps(DATA_CONFIG)

        arguments_part = [['--project_name', project_name],
                          ['--configuration', config_arg],
                          ['--data_name', data_name],
                          ['--task', args.task],
                          ['--data_configuration', data_arg],
                          ['--model_name', model_name]
                          ]

        arguments = reduce(lambda v, w: v + w, arguments_part)

        # collect package
        logger.info(
            'Checking if packages needed by ML engine are available')

        packages = {
            p: 'gs://{bucket_name}/ai_platform_template_dir/package/{package}'.format(
                bucket_name=bucket_name,
                package=p
            ) for p in os.listdir(os.path.join(project_dir_path,
                                               'package')
                                  )
        }
        logger.debug("package URIs: %s " % list(packages.values()))
        sorted_packages_uris = sorted(list(packages.values()))

        gs_interface.load_package_to_storage(packages=packages,
                                             bucket_name=bucket_name,
                                             parent_path=project_dir_path
                                             )

        # setting ML engine machine
        cloud_ml_conf_relative_path = os.path.join(project_dir_path, 'configuration', 'cloud_ml_parameters.yaml')
        cloud_ml_path = os.path.join(project_dir_path,
                                     cloud_ml_conf_relative_path
                                     )
        with open(cloud_ml_path, 'r') as f:
            cloud_ml_param = yaml.safe_load(f)

        # setting up the job
        logger.info('Start computation in the cloud')
        ml_engine_service = discovery.build(serviceName='ml',
                                            version='v1',
                                            credentials=credentials,
                                            cache_discovery=False)

        job_parent = "projects/{project}".format(
            project=project_name)

        job_id = get_job_id(model_name='titanic_survival_model')

        job_body = get_job_body(
            cloud_ml_param=cloud_ml_param,
            job_id=job_id,
            module_to_call='models.classifier',
            arguments=arguments,
            sorted_packages_uris=sorted_packages_uris)

        logger.info("job_body: %s" % job_body)
        logger.info("job_parent: %s" % job_parent)
        logger.info("creating a job ml: %s" % job_id)

        job_ml = ml_engine_service.projects().jobs().create(
            parent=job_parent, body=job_body).execute()

        time.sleep(5)

        # check job status
        try:
            succeeded_job = is_success(ml_engine_service=ml_engine_service,
                                       project_id=project_name,
                                       job_id=job_id)
            if succeeded_job:
                logger.info('Training job done')
            else:
                logger.error('Training job failed')
                sys.exit(1)
        except Exception as e:
            logger.error(e)
            sys.exit(1)

        logger.info(
            "[FINISHED] Job {job_id} completed task {task}".format(
                job_id=job_id, task=args.task)
        )

        ml_units_used += get_consumed_ml_units(
            ml_engine_service=ml_engine_service,
            project_id=project_name,
            job_id=job_id)
        # 1 ml-unit = 0.54$

    else:
        logger.info("Run in local")
        # here depending on task
        if args.task == 'train':
            train_model_in_local(
                gs_interface=gs_interface,
                gs_dir_path=config['google_gcs'].get('directory_name'),
                bucket_name=config['google_gcs'].get('bucket_name'),
                local_dir_path=config.get('local_dir_path', "tmp"),
                train_name=data_name,
                data_configuration=DATA_CONFIG,
                model_name=model_name
            )
        elif args.task == 'predict':
            predict_in_local(gs_interface=gs_interface,
                             gs_dir_path=config['google_gcs'].get('directory_name'),
                             bucket_name=config['google_gcs'].get('bucket_name'),
                             local_dir_path=config.get('local_dir_path', "tmp"),
                             predict_name=data_name,
                             data_configuration=DATA_CONFIG,
                             model_name=model_name
                             )

        else:
            pass

    logger.info(f"Task {args.task} is successfully completed. It has consumed {ml_units_used} ML units")
    # see https://cloud.google.com/vertex-ai/pricing#training for the pricing
