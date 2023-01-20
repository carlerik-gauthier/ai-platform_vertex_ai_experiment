import datetime
import logging
import time
from google.oauth2 import service_account
import google.auth as ga

logger = logging.getLogger(__name__)


def is_success(ml_engine_service, project_id, job_id):
    """
    Doc:
    https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#State

    :param ml_engine_service: discovery.build, the container to run the job
    :param project_id: str, the name of the project id
    :param job_id: str, the name of the job id
    :return:
    """

    wait = 60  # seconds
    no_time_diff = datetime.timedelta(seconds=10)
    timeout_preparing = datetime.timedelta(seconds=900)
    timeout_running = datetime.timedelta(hours=24)

    api_call_time = datetime.datetime.now()
    api_job_name = "projects/{project_id}/jobs/{job_name}".format(
        project_id=project_id, job_name=job_id)
    job_description = ml_engine_service.projects().jobs().get(
        name=api_job_name).execute()

    while job_description["state"] not in ["SUCCEEDED", "FAILED", "CANCELLED"]:
        # check here the PREPARING and RUNNING state to detect the
        # abnormalities of ML Engine service

        threshold = no_time_diff if job_description["state"] == "PREPARING" \
            else timeout_running
        threshold += timeout_preparing
        delta = process_job(ml_engine_service=ml_engine_service,
                            state=job_description["state"],
                            threshold=threshold,
                            api_call_time=api_call_time,
                            job_id=job_id,
                            api_job_name=api_job_name)

        logger.info("""
        [ML] NEXT UPDATE for job {job_id} IN {wait} seconds 
        ({delta} seconds ELAPSED IN {state} STAGE)
        """.format(job_id=job_id, wait=wait, delta=delta,
                   state=job_description["state"])
                    )
        job_description = ml_engine_service.projects().jobs().get(
            name=api_job_name).execute()
        time.sleep(wait)

    logger.info("Job '%s' is completed" % job_id)

    # Check the job state
    if job_description["state"] == "SUCCEEDED":
        logger.info("Job '%s' succeeded!" % job_id)
        return True

    else:
        logger.error(job_description["errorMessage"])
        return False


def get_consumed_ml_units(ml_engine_service, project_id, job_id):
    api_job_name = "projects/{project_id}/jobs/{job_name}".format(
        project_id=project_id, job_name=job_id)
    job_description = ml_engine_service.projects().jobs().get(
        name=api_job_name).execute()

    return job_description['trainingOutput']['consumedMLUnits']


def process_job(ml_engine_service,
                state,
                threshold,
                api_call_time,
                job_id,
                api_job_name):
    """
    :param ml_engine_service: discovery.build, the container to run the job
    :param state: str: state of the job
    :param threshold : int: threshold above which job is killed
    :param api_call_time: datetime: datetime from api call
    :param job_id: str:
    :param api_job_name: str
    :return: delta : int: time spent since call to the api
    """
    delta = datetime.datetime.now() - api_call_time
    if delta > threshold:
        logger.error("""
                    [ML] {state} stage timeout after {time} seconds 
                    --> CANCEL job {job_id}
                    """.format(state=state, time=delta.seconds,
                               job_id=job_id)
                     )

        ml_engine_service.projects().jobs().cancel(name=api_job_name,
                                                   body={}).execute()

        raise Exception

    return delta.seconds


def get_credentials(credentials_json_file=None):
    """
    :param credentials_json_file: dict
    :return: Google credential
    """
    credentials, _ = ga.default()
    if credentials_json_file is not None:
        credentials = service_account.Credentials.from_service_account_file(
            credentials_json_file)

    return credentials


def get_job_body(cloud_ml_param: dict,
                 job_id: str,
                 module_to_call: str,
                 arguments: list,
                 sorted_packages_uris: list):

    mlmachine_size = cloud_ml_param.get('mlmachine_size')
    typology_machine = cloud_ml_param.get('typology_machine')[mlmachine_size]
    # https://cloud.google.com/ai-platform/training/docs/training-jobs
    job_body = {'trainingInput':
                {'pythonVersion': cloud_ml_param.get('ml_pythonVersion'),
                 'runtimeVersion': cloud_ml_param.get('ml_runtimeVersion'),
                 'scaleTier': typology_machine['ml_scaleTier_train'],
                 'region': cloud_ml_param.get('ml_region'),
                 'pythonModule': module_to_call,
                 'args': arguments,
                 'packageUris': sorted_packages_uris,
                 'masterType': typology_machine['ml_masterType'],
                 'workerType': typology_machine['ml_workerType'],
                 'workerCount': typology_machine['ml_workerCount'],
                 'parameterServerCount':
                     typology_machine['ml_parameterServerCount'],
                 'parameterServerType':
                     typology_machine['ml_parameterServerType']
                 },
                'jobId': job_id}

    return job_body


def get_job_id(model_name: str):
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    job_id = "job_{model_name}_{time}".format(model_name=model_name,
                                              time=now_str)
    return job_id
