# https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/prediction/custom_prediction_routines/SDK_Custom_Predict_and_Handler_SDK_Integration.ipynb
# https://codelabs.developers.google.com/vertex-cpr-sklearn#5

# https://github.com/googleapis/python-aiplatform/blob/custom-prediction-routine/google/cloud/aiplatform/prediction/sklearn/predictor.py
# https://cloud.google.com/vertex-ai/docs/predictions/custom-prediction-routines
# https://cloud.google.com/ai-platform/prediction/docs/custom-prediction-routines#predictor-class

import pickle
import pandas as pd
import numpy as np
import joblib
import json
import os
import logging

from google.cloud import storage
from google.cloud.aiplatform.prediction.sklearn.predictor import SklearnPredictor
from google.cloud.aiplatform.utils import prediction_utils

LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=LOGLEVEL)
logger = logging.getLogger()

# Our model is a scikit-learn model


class CprPredictor(SklearnPredictor):
    # https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/prediction/custom_prediction_routines/SDK_Custom_Predict_and_Handler_SDK_Integration.ipynb
    # https://cloud.google.com/ai-platform/prediction/docs/custom-prediction-routines#scikit-learn
    # https://cloud.google.com/vertex-ai/docs/predictions/custom-prediction-routines
    # https://github.com/googleapis/python-aiplatform/blob/main/google/cloud/aiplatform/prediction/sklearn/predictor.py
    COLUMNS = ["PassengerId", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]
    TYPES = {"Pclass": int, "Age": float, "SibSp": int, "Parch": int, "Fare": float}

    def __init__(self):
        self._preprocessor = None
        self._model = None
        super().__init__()

    def load(self, artifacts_uri: str) -> None:
        """Loads the sklearn pipeline and preprocessing artifact.
        artifacts_uri (str):
                Required. The value of the environment variable AIP_STORAGE_URI


        Artifact_uri is gs://caip-tenant-c3fd2645-597a-44af-a7c3-c8a51be9a5a0/8918094832426024960/artifacts
        gs://caip-tenant-dba91617-3854-46d9-b811-c054a154893a/6576785976146788352/artifacts
        Artifact_uri is gs://caip-tenant-cbf5571b-8860-436a-bec7-ed8da970597b/5166596342826401792/artifacts
        """
        # prediction_utils.download_model_artifacts(artifacts_uri)
        super().load(artifacts_uri)

        logger.info(f"---- Artifact_uri is {artifacts_uri}")
        with open("preprocessor.pkl", "rb") as f:
            preprocessor = pickle.load(f)

        self._model = joblib.load("model.joblib")
        self._preprocessor = preprocessor

    def preprocess(self, prediction_input: dict) -> np.ndarray:
        """Performs preprocessing by checking if clarity feature is in abbreviated form."""

        # inputs = super().preprocess(prediction_input)
        instances = prediction_input["instances"]
        inputs_df = pd.DataFrame(data=np.array(instances),
                                 columns=self.COLUMNS)
        inputs_df = inputs_df.astype(self.TYPES)
        # features = self._data_config.get("features")
        # features = dict() if features is None else features
        inputs_preproc = self._preprocessor.preprocess(
            df=inputs_df,
            # age_col=features.get('age_col'),
            # gender_col=features.get('gender_col'),
            # fixed_columns=features.get('fixed')
            )
        inputs = inputs_preproc.values
        # below is irrelevant in our example
        # for sample in inputs:
        #    if sample[3] not in self._preprocessor.values():
        #        sample[3] = self._preprocessor[sample[3]]
        return np.asarray(inputs)

    # def postprocess(self, prediction_results: np.ndarray) -> dict:
    #    """ Performs postprocessing. Useless in this example """
    #    return prediction_results
