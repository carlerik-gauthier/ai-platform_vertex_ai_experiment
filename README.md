# Using Vertex AI with CustomPythonPackageJob

# TODO :
- deployment + prediction (batch?)
- Preprocessing before or during prediction steps
- CustomPythonPackageTrainingJob
- full custom : avec création d'une propre image docker à push sur registry

v1 l'actuel, le v2 selon predict preprocess
le v3 customPythonPackageJob
le v4 le full custom

# Objectif
https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform
https://cloud.google.com/artifact-registry/docs/python/manage-packages#gcloud
https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/custom/SDK_Custom_Container_Prediction.ipynb


Le but est de fournir un template de code permettant de lancer un entraînement de modèle ML avec Vertex AI en utilisant
des images pre-built. L'approche "easy" est montrée

# Data
Pour ce template, nous utilisons le jeu de données 'Titanic dataset' disponible sur Kaggle.

Il n'y a pas d'optimisation du modèle de prédiction de survie. Il est juste là pour illustrer.

# Steps
- https://cloud.google.com/artifact-registry/docs/repositories/create-repos?hl=fr#gcloud
- https://cloud.google.com/vertex-ai/docs/start/ai-platform-users 
- https://cloud.google.com/vertex-ai/docs/predictions/custom-prediction-routines
- https://cloud.google.com/vertex-ai/docs/training/create-python-pre-built-container
- https://cloud.google.com/vertex-ai/docs/training/create-custom-job#create_custom_job-python THIS ONE
- https://cloud.google.com/vertex-ai/docs/training/create-training-pipeline THIS ONE TOO
- https://cloud.google.com/vertex-ai/docs/training/pre-built-containers
- https://cloud.google.com/vertex-ai/docs/training/custom-training
- https://cloud.google.com/vertex-ai/docs/training/code-requirements
- https://cloud.google.com/vertex-ai/docs/start/migrating-applications?hl=fr
- https://cloud.google.com/vertex-ai/docs/start/migrating-to-vertex-ai?hl=fr#ai-platform
- https://cloud.google.com/vertex-ai/docs/start/client-libraries
--> pip install google-cloud-aiplatform // https://github.com/googleapis/python-aiplatform/ HERE
- https://cloud.google.com/vertex-ai/docs/predictions/use-custom-container#aiplatform_upload_model_highlight_container-python and HERE
- https://www.youtube.com/watch?v=VRQXIiNLdAk&t=516s THIS ONE
- https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/bigquery_ml/bqml-online-prediction.ipynb?utm_medium=email&utm_source=burgersandfries&utm_campaign=VertexAItoBQ4&utm_content=en
- https://cloud.google.com/blog/topics/developers-practitioners/using-vertex-ai-rapid-model-prototyping-and-deployment?hl=en
- https://codelabs.developers.google.com/vertex-pipelines-intro#0
- https://cloud.google.com/vertex-ai/docs/predictions/custom-prediction-routines 
- https://medium.com/google-cloud/how-to-train-ml-models-with-vertex-ai-training-f9046bfbcfab 
- https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/explainable_ai/sdk_custom_image_classification_online_explain.ipynb

https://cloud.google.com/vertex-ai/docs/predictions/custom-prediction-routines
https://github.com/googleapis/python-aiplatform/tree/custom-prediction-routine/google/cloud/aiplatform/prediction
https://github.com/GoogleCloudPlatform/mlops-on-gcp/blob/master/on_demand/kfp-caip-sklearn/lab-01-caip-containers/lab-01.ipynb
https://codelabs.developers.google.com/vertex-cpr-sklearn#5
https://stackoverflow.com/questions/35153902/find-the-list-of-google-container-registry-public-images
https://blog.searce.com/deploy-your-own-custom-ml-on-vertex-ai-using-gcp-console-e3c52f7da2b
https://cloud.google.com/vertex-ai/docs/training/create-custom-container
https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/prediction/custom_prediction_routines/SDK_Custom_Predict_and_Handler_SDK_Integration.ipynb
https://blog.ml6.eu/vertex-ai-is-all-you-need-599ffc9473fd
https://towardsdatascience.com/how-to-set-up-custom-vertex-ai-pipelines-step-by-step-467487f81cad
https://supertype.ai/notes/deploying-machine-learning-models-with-vertex-ai-on-google-cloud-platform/
https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/custom/SDK_Custom_Container_Prediction.ipynb
BELOW : might change

dataset management
https://cloud.google.com/vertex-ai/docs/training/using-managed-datasets#sentiment-analysis

# Vertex AI avec des images pre-built ET des CustomPythonPackageTrainingJob

## 1. La data dans Storage, why ? 
## 2. Entraîner le modèle et sauver le modèle dans Storage depuis le local
## 3. Packaging
## 4. Faire une opération avec AI-Platform

# Pré-requis
**0.** Activer les APIs Vertex AI et Artifact Registry

**A.** Avoir les credentials GCP; en particulier pour Storage.  

**B.** Dans les arguments du main.py, définir une variable qui permet de savoir s'il faut lancer les calculs avec 
Vertex AI ou le faire en local. Par ailleurs, il existe une notion de version pour l'endpoint

Ici, ces arguments s'appellent env et endpoint_version.

# 1. La data dans Storage, why ? 
### A. Why ?
Il est important de rappeler que les machines AI-Platform ont des droits GCP liés aux credentials utilisés pour les 
jobs sur la AI-Platform.

Pour des mesures de sécurité, elles n'ont pas de droits d'écritures sur BigQuery.

Suivant les règles édictées par Google, il est préfèrable d'interagir **uniquement** avec Google Storage : 
- *input rules :* https://cloud.google.com/ai-platform/training/docs/overview#input_data
- *output rules :* https://cloud.google.com/ai-platform/training/docs/overview#output_data

### B. Setup
Avant de commencer les manipulations de la Data, il est nécessaire, pour plus de clarté, de créer un directory 
dans le bucket Storage dédié. La donnée qui devra être consommée par la tâche exécutée au sein de la AI-Platform
devra y être déposée.

À noter également que le modèle ML/DL entraînée sera également sauvegardé dans ce directory.

Pour cet exemple, le bucket est *dmp-y-test-bucket* et le directory de ce projet est 
*ai_platform_template_dir*.


# 2. Entraîner le modèle et sauver le modèle dans Storage depuis le local
### A. L'interface principal
Avant toute opération, il est important de tester les tâches destinées au cloud en local et dans un environnement 
virtuel propre, i.e. qui ne contient que les librairies spécifiée dans requirements.txt, et les dépendances nécessaires.
Pour cela, l'argument env doit être 'local' lors de l'exécution de l'interface principale qu'est src/main.py .

Ainsi, la ligne de commande est 
```
python3 {CODE_PATH}/src/main.py --task {TASK} --config {PATH_TO_CONFIGURATION_FILE} --env local
```

Dans cet exemple, il y a 2 tâches : train & predict.

La première vérification est de s'assurer que le code ne crashe pas.

Une fois certain que le code fonctionne et dans le cadre de tâches où de la data transformée ou un modèle doit être 
produit en output, vérifier que l'output se trouve bien là il est supposé être.

### B. L'exécutable lancée pour l'entraînement du modèle
Il s'agit du fichier qui sera lancée par la AI-Platform. Il est le point d'entrée pour les opérations de calculs à faire 
dans le Cloud. Lors de calculs en local, le code principal importe les fonctions nécessaires se trouvant dans ce module.

Le but de ce point est de s'assurer que la récupération des arguments et que les bonnes informations sont transmises aux 
fonctions de calculs.

Ici, il s'agit du fichier src/models/classifier.py

**NB:** json.dumps transforme un dictionnaire en string : json.loads({dict}) = ‘{dict}’; 
alors que json.loads procède à l'opération inverse : json.loads('{dict}') = {dict}.            

# 3. Packaging
Dans l'approche "image pre-built" avec l'approche simple, il n'y a pas de packaging spécifiques à faire. Tout est géré
par Google car Vertex AI a pour but de centraliser toutes ces opérations afin de rendre le déploiement le plus simple 
possible.

Toutefois, il y a certaines étapes à suivre :
    1. Data Registry en fonction du type de données. Ici, il s'agit de données tabulaire
    2. Construire un modèle qui sera accessible depuis le Model Registry. L'entraînement du modèle peut être suivi dans 
l'onglet ...
    3. Déployer le modèle sur un Endpoint. Il s'agit du point d'entrée utilisé pour faire les prédictions. Le Endpoint
est visible sur la Endpoint Registry.

**Attention** : Un Endpoint sera toujours facturé même lorsqu'il n'est pas utilisé.

# 4.A. Faire un entraînement avec Vertex AI
Nous voilà prêt pour passer dans le Cloud.

Pour accéder à la console Vertex AI il suffit d’aller sur l’onglet des différentes fonctionnalités de la GCP et 
cliquer sur Vertex AI. 

Pour lancer une opération d'entraînement, il suffit de lancer la même commande que pour le local, mais en remplaçant le contenu de 
l’argument d’environnement par la valeur nécessaire. C’est la seule modification à faire. 
```
python3 {CODE_PATH}/src/main.py --task {TASK} --config {PATH_TO_CONFIGURATION_FILE} --env cloud
```
# 4.B. Faire une prédiction avec Vertex AI
Dans le cadre de prédictions avec un Endpoint, la data **doit être** préprocessée en amont. Il n'est pas possible de le
faire en cours de prédiction.

Pour une prédiction avec l'Endpoint, il faut procéder aux étapes suivantes : 
    1.
    2.
    3.

# Pricing 
https://cloud.google.com/vertex-ai/pricing#europe


# NEW STEPS 
pip install --upgrade pip setuptools wheel
pip install twine keyrings.google-artifactregistry-auth


### create a python repository if wanted
gcloud artifacts repositories create <your_repository_name> --repository-format=python 
--location=<your_repository_location>
--description="<your_repository_description>"

### build the package
python setup.py bdist_wheel

cp dist/{package_name}-{package_version}-py3-none-any.whl ../package/

### Deploy the sample python package to artifact registry (dist has to be a subfolder from current position)
twine upload --repository-url https://<your_repository_location>-python.pkg.dev/<your_gcp_project>/<your_repository_name>/ dist/*
