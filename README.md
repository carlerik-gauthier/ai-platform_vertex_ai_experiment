# Custom Predict Routines branch
# Objectif

Le but est de fournir un template de code permettant de lancer un entraînement de modèle ML avec Vertex AI en utilisant
des images pre-built. L'approche "easy" est montrée

# Data
Pour ce template, nous utilisons le jeu de données 'Titanic dataset' disponible sur Kaggle.

Il n'y a pas d'optimisation du modèle de prédiction de survie. Il est juste là pour illustrer.

# Documentation
- https://cloud.google.com/artifact-registry/docs/repositories/create-repos?hl=fr#gcloud
- https://cloud.google.com/vertex-ai/docs/training/exporting-model-artifacts?hl=fr#scikit-learn
- https://cloud.google.com/vertex-ai/docs/training/code-requirements 
- 
- https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/prediction/custom_prediction_routines/SDK_Custom_Predict_and_Handler_SDK_Integration.ipynb
- https://github.com/GoogleCloudPlatform/cloudml-samples/blob/main/notebooks/scikit-learn/custom-prediction-routine-scikit-learn.ipynb
- https://codelabs.developers.google.com/vertex-cpr-sklearn#5
- https://cloud.google.com/vertex-ai/docs/predictions/custom-prediction-routines


--> pip install google-cloud-aiplatform : https://github.com/googleapis/python-aiplatform/ 

-- Vertex Pipeline
- https://cloud.google.com/blog/topics/developers-practitioners/using-vertex-ai-rapid-model-prototyping-and-deployment?hl=en
- https://codelabs.developers.google.com/vertex-pipelines-intro#0
- https://towardsdatascience.com/how-to-set-up-custom-vertex-ai-pipelines-step-by-step-467487f81cad

-- other
- https://github.com/googleapis/python-aiplatform/tree/custom-prediction-routine/google/cloud/aiplatform/prediction
- https://codelabs.developers.google.com/vertex-cpr-sklearn#5



# Vertex AI avec des images pre-built : Custom Predict Routines

https://aster.cloud/2022/09/02/simplify-model-serving-with-custom-prediction-routines-on-vertex-ai/
https://cloud.google.com/python/docs/reference/aiplatform/1.19.1/google.cloud.aiplatform.prediction.LocalModel#google_cloud_aiplatform_prediction_LocalModel_build_cpr_model

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


Pour cet exemple, le bucket est *dmp-y-test_vertex_ai_bucket* et le directory de ce projet est 
*ai_platform_template_dir_exp_2*.

# 1. Le prédicateur
### A. Why ?

Pour cet exemple, le bucket est *dmp-y-test-bucket* et le directory de ce projet est 
*ai_platform_template_dir*.

# 2. Les changements dans le main.py
         

# 3. Factorisation du projet 


# 4. Vérifier sur Vertex AI
Nous voilà prêt pour passer dans le Cloud.

Pour accéder à la console Vertex AI il suffit d’aller sur l’onglet des différentes fonctionnalités de la GCP et 
cliquer sur Vertex AI. 

Pour lancer une opération d'entraînement, il suffit de lancer la même commande que pour le local, mais en remplaçant le contenu de 
l’argument d’environnement par la valeur nécessaire. C’est la seule modification à faire. 
```
python3 {CODE_PATH}/src/main.py --task {TASK} --config {PATH_TO_CONFIGURATION_FILE} --env cloud
```
# 5. Et les prédictions ?
Dans le cadre de prédictions avec un Endpoint, la data **doit être** préprocessée en amont. Il n'est pas possible de le
faire en cours de prédiction.

Pour une prédiction avec l'Endpoint, il faut procéder aux étapes suivantes : 
    1.
    2.
    3.

# Pricing 
https://cloud.google.com/vertex-ai/pricing#europe


```pip install google-cloud-aiplatform[prediction]>=1.16.0```
