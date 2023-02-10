# Custom Predict Routines branch 
### Vertex AI avec des images pre-built : Custom Predict Routines

# Documentation
- https://cloud.google.com/artifact-registry/docs/repositories/create-repos?hl=fr#gcloud
- https://cloud.google.com/vertex-ai/docs/training/exporting-model-artifacts?hl=fr#scikit-learn
- https://cloud.google.com/vertex-ai/docs/training/code-requirements
- https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/prediction/custom_prediction_routines/SDK_Custom_Predict_and_Handler_SDK_Integration.ipynb
- https://github.com/GoogleCloudPlatform/cloudml-samples/blob/main/notebooks/scikit-learn/custom-prediction-routine-scikit-learn.ipynb
- https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/prediction/custom_prediction_routines/SDK_Pytorch_Custom_Predict.ipynb
- https://codelabs.developers.google.com/vertex-cpr-sklearn#5
- https://cloud.google.com/vertex-ai/docs/predictions/custom-prediction-routines
- https://cloud.google.com/vertex-ai/docs/predictions/migrate-cpr?hl=fr
- https://github.com/googleapis/python-aiplatform/blob/main/google/cloud/aiplatform/prediction/sklearn/predictor.py


- pip install google-cloud-aiplatform : https://github.com/googleapis/python-aiplatform/ 

-- Vertex Pipeline
- https://cloud.google.com/blog/topics/developers-practitioners/using-vertex-ai-rapid-model-prototyping-and-deployment?hl=en
- https://codelabs.developers.google.com/vertex-pipelines-intro#0
- https://towardsdatascience.com/how-to-set-up-custom-vertex-ai-pipelines-step-by-step-467487f81cad

-- other related cpr references
- https://github.com/googleapis/python-aiplatform/tree/custom-prediction-routine/google/cloud/aiplatform/prediction
- https://aster.cloud/2022/09/02/simplify-model-serving-with-custom-prediction-routines-on-vertex-ai/
- https://cloud.google.com/python/docs/reference/aiplatform/1.19.1/google.cloud.aiplatform.prediction.LocalModel#google_cloud_aiplatform_prediction_LocalModel_build_cpr_model
- https://blog.ml6.eu/deploy-ml-models-on-vertex-ai-using-custom-containers-c00f57efdc3c

# Objectif

Le but est de fournir un template de code permettant de personnaliser les routines de prédictions afin de préprocesser l'input 
durant la phase de prédiction qui est déclenchée par l'appel au Endpoint ou par le modèle enregistrée. 
En effet, par défaut, le préprocessing doit être fait en amont, ce qui n'est pas, selon la situation, la situation idéale. 

# Data
Pour ce template, nous utilisons le jeu de données 'Titanic dataset' disponible sur Kaggle.

Il n'y a pas d'optimisation du modèle de prédiction de survie. Il est juste là pour illustrer.

# Steps
## 1. Écrire un prédicteur et, optionnellement, un handler
## 2. Les changements dans le main.py
## 3. Factorisation du projet 
## 4. Vérifier sur Vertex AI
## 5. Et les prédictions ?

# Pré-requis
**0.** Activer les APIs Vertex AI et Artifact Registry

**A.** Avoir les credentials GCP; en particulier pour Storage.  

**B.** Dans les arguments du main.py, définir une variable qui permet de savoir s'il faut lancer les calculs avec 
Vertex AI ou le faire en local. Par ailleurs, il existe une notion de version pour l'endpoint

Ici, ces arguments s'appellent env et endpoint_version.

**C.** Les artifacts du modèle et du préprocesseur existent déjà et sont mis dans un bucket Storage

**D.** Installer google-cloud-aiplatform[prediction]>=1.16.0 ```pip install google-cloud-aiplatform[prediction]>=1.16.0```

Pour cet exemple, le bucket est *dmp-y-test_vertex_ai_bucket* et le directory de ce projet est 
*ai_platform_template_dir_exp_2*.

De plus, nous **supposons** qu'un modèle (et son préprocesseur) est déjà entrainé et que ses artefacts sont disponible dans GCS. 
Dans cet exemple, ils se trouvent dans *dmp-y-test_vertex_ai_bucket/ai_platform_template_dir_exp_2/cpr_model_artifact*

# 1. Le prédicateur
### A. Why ?
Comme mentionné dans les références :
- https://cloud.google.com/vertex-ai/docs/predictions/custom-prediction-routines?hl=fr
- https://cloud.google.com/vertex-ai/docs/predictions/migrate-cpr?hl=fr
- https://github.com/GoogleCloudPlatform/cloudml-samples/blob/main/notebooks/scikit-learn/custom-prediction-routine-scikit-learn.ipynb
- https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/prediction/custom_prediction_routines/SDK_Custom_Predict_and_Handler_SDK_Integration.ipynb

il est obligatoire de créer un objet Predicator qui hérite (directement ou non) de l'objet abstrait *Predictor* se trouvant 
dans google.cloud.aiplatform.prediction.predictor. Une implémentation pour des modèles scikit-learn existe déjà et le
détail peut être trouvé dans https://github.com/googleapis/python-aiplatform/blob/main/google/cloud/aiplatform/prediction/sklearn/predictor.py .

Les opérations de préprocessing doivent être décrites dans la méthode **preprocessing**.

Le codelab https://codelabs.developers.google.com/vertex-cpr-sklearn#5 fournit de bonnes indications
# 2. Les changements dans le main.py

Par rapport à la branche *vertex-ai*, la différence principale réside dans la partie "modèle". 
Ici, il s'agit de générer un image Docker au sein de "Artifact Registry" au moyen d'un **LocalModel** 
(from google.cloud.aiplatform.prediction import LocalModel). 

Une fois que nous disposons d'un modèle local, il s'agit de "matérialiser" le modèle avant de le déployer.  

# 3. Factorisation du projet 
L'utilisation d'une CPR implique la création par Vertex AI d'une image Docker enregistrée dans la Register Artifact. La 
conséquence est que tout le code de création et d'entraînement de modèles et des tâches de préprocessing doivent être dans 
un folder spécifique. Ici, il s'appelle *cpr_dir*.

Afin de disposer d'artefact de préprocessing, les différentes étapes doivent être encapsulées dans un unique objet, voir
cpr_dir/preprocessing/vertex_titanic_preprocessor.py.

Quant à l'arborescence de *cpr_dir*, il doit ressembler à 
```bash
├── __init__.py
├── predictor.py
├── handler.py
└── requirements.txt
├── gcp_interface
│ ├── ...
├── models
│ ├── ...
├── preprocessing
│ ├── ...
└── requirements.txt
```
En particulier, au niveau le plus haut, nous devons avoir :
- requirements.txt : il contient les packages à installer dans l'image contruite.
- predictor.py : ce fichier contient le code nécessaire pour exprimer la logique et les opérations à mener dans la procédure de prédiction
Une implémentation pour des modèles scikit-learn se trouve dans https://github.com/googleapis/python-aiplatform/blob/main/google/cloud/aiplatform/prediction/sklearn/predictor.py
- (Optionnel) handler.py : Il contient le code pour personnaliser la logique liée au serveur web. 


La création à proprement parlé de l'image Docker est faite dans *vertex_ai_experiment/containers.py* grâce au bout de code
```
local_model = LocalModel.build_cpr_model(
        src_dir=path_to_source_dir, # le folder contenant le predicteur
        output_image_uri=f"{location}-docker.pkg.dev/{project_id}/{repository}/{image}",
        predictor=CprPredictor, # le prédicteur défini
        handler=CprHandler, # le handler éventuellement défini
        requirements_path=path_to_requirement, # le chemin vers le fichier de requirements située dans src_dir
        extra_packages=extra_packages)
```
# 4. Vérifier sur Vertex AI
Comme pour les images pre_built
# 5. Et les prédictions ?
Comme pour les images pre_built

# Pricing 
https://cloud.google.com/vertex-ai/pricing#europe

