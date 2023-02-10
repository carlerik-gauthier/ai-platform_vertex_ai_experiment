# Vertex AI avec des images pre-built : approche la plus simple

# Documentation 
- https://cloud.google.com/docs?hl=fr
- https://cloud.google.com/vertex-ai/docs/training/custom-training
- https://cloud.google.com/vertex-ai/docs/training/create-python-pre-built-container
- https://cloud.google.com/vertex-ai/docs/training/create-custom-job#create_custom_job-python 
- https://cloud.google.com/vertex-ai/docs/training/create-training-pipeline#script 
- https://cloud.google.com/vertex-ai/docs/training/pre-built-containers
- https://cloud.google.com/vertex-ai/docs/predictions/pre-built-containers?hl=fr

- google-cloud-aiplatform : https://github.com/googleapis/python-aiplatform/
- https://cloud.google.com/python/docs/reference/aiplatform/latest#google.cloud.aiplatform.CustomContainerTrainingJob
- https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/explainable_ai/sdk_custom_image_classification_online_explain.ipynb
- https://cloud.google.com/vertex-ai/docs/tutorials/tabular-bq-prediction/create-training-script?hl=fr

other
- https://cloud.google.com/vertex-ai/docs/start/ai-platform-users
- https://cloud.google.com/vertex-ai/docs/training/code-requirements
- https://cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements
- https://cloud.google.com/vertex-ai/docs/start/migrating-to-vertex-ai?hl=fr#migration-info

# Objectif
Le but est de fournir un template de code permettant de lancer un entraînement de modèle ML avec Vertex AI en utilisant
des images pre-built. L'approche "easy" est montrée.

# Data
Pour ce template, nous utilisons le jeu de données 'Titanic dataset' disponible sur Kaggle.

Il n'y a pas d'optimisation du modèle de prédiction de survie. Il est juste là pour illustrer.

# Steps
## 1. Principes
## 2. Ce qui change avec AI-Platform
## 3. Entraîner et prédire

# Pré-requis
**0.** Activer les APIs Vertex AI : https://cloud.google.com/vertex-ai/docs/start/client-libraries

**A.** Avoir les credentials GCP; en particulier pour Storage.  S'assurer que l'emplacement du bucket utilisé est 
de type **Region**. En effet, Vertex AI n'admet pas l'utilisation d'un type "Multi-region" car il gère l'infrastructure.

**B.** Dans les arguments du main.py, définir une variable qui permet de savoir s'il faut lancer les calculs avec 
Vertex AI ou le faire en local. Par ailleurs, il existe une notion de version pour l'endpoint

Ici, ces arguments s'appellent env et endpoint_version.

# 1. Principes
L'approche "image pre-built" est l'approche simple et il n'y a pas de packaging spécifique à faire. Tout est géré
par Google car Vertex AI a pour but de centraliser toutes ces opérations afin de rendre le déploiement le plus simple 
possible.

Toutefois, il y a certaines étapes à suivre :

    1. Les données doivent se trouver dans le **Data Registry** en fonction du type de données. Cela se trouve dans Vertex AI >> Ensemble de données
(le logo carré avec 4 points dedans). Ici, il s'agit de données tabulaire. Toutefois, des données textuelles, d'images 
et de vidéos sont également admissibles.

    2. Construire un modèle qui sera accessible depuis le **Model Registry**. L'entraînement du modèle peut être suivi dans 
l'onglet Vertex AI >> Model Registry (le logo en forme d'ampoule.)

    3. (Optionnel) Déployer le modèle sur un *Endpoint*. Il s'agit du point d'entrée utilisé pour faire les prédictions. Le Endpoint
est visible sur le **Endpoint Registry** (logo en forme d'ampoule émitrice). Cette étape est nécessaire pour des prédictions en temps réel.
Néanmoins, une version de prédiction par lot est possible. Auquel cas, il n'est pas nécessaire de faire un déploiement sur un Endpoint. 
Il suffit de récupérer le modèle et l'utiliser dans le code.

**Attention** : Un Endpoint sera toujours facturé même lorsqu'il n'est pas utilisé.

# 2. Ce qui change avec AI-Platform

Premièrement, il est important de savoir que Vertex AI définit des variables d'environnements :   
 1. AIP_DATA_FORMAT : format d'exportation de vl'ensemble de données. Les valeurs possibles incluent jsonl, csv ou bigquery
 2. AIP_TRAINING_DATA_URI : URI BigQuery des données d'entraînement ou URI Cloud Storage du fichier de données d'entraînement.
 3. AIP_VALIDATION_DATA_URI : URI BigQuery des données de validation ou URI Cloud Storage du fichier de données de validation
 4. AIP_TEST_DATA_URI : URI BigQuery des données de test ou URI Cloud Storage du fichier de données de test.
 5. AIP_MODEL_DIR : URI Cloud Storage d'un répertoire destiné à enregistrer les artefacts de modèles.

Pour davantage de détails, voir 
- https://cloud.google.com/python/docs/reference/aiplatform/latest#google.cloud.aiplatform.CustomContainerTrainingJob
- https://cloud.google.com/vertex-ai/docs/training/using-managed-datasets#access_a_dataset_from_your_training_application
- https://cloud.google.com/vertex-ai/docs/training/code-requirements?hl=fr#environment-variables

L'utilisation de cette approche implique de choisir :
- 1 image pré-construite pour l'entraînement, à choisir dans la liste : https://cloud.google.com/vertex-ai/docs/training/pre-built-containers
- 1 image pré-construite pour les prédictions, à choisir dans la liste : https://cloud.google.com/vertex-ai/docs/predictions/pre-built-containers?hl=fr

Dans cet exemple, elles sont renseignées dans le fichier de configuration. 

Par rapport à la création d'un job AI-Platform, il y a 3 changements principaux :

A. Aucun packaging de code n'est nécessaire.

B. Le script en charge de l'entraînement du modèle doit être "autonome" vis-à-vis de la structure du projet. Seul 
l'import de packages disponible dans PyPI est admis. Ceux-ci doivent être compatible avec les images choisies. 
En particulier, cela implique que tout le préprocessing, la création du modèle et son entraînement doivent être écrit dans 
un **unique script**. 

À noter que les variables environnementales de Vertex-AI ci-dessus doivent être utilisées pour l'enregistrement du modèle
et la récupération des données.

Dans cet exemple, le script est ```src/models/vertex_classifier.py```

C. Le script ```main.py``` est modifié dans la partie d'entraînement et de prédiction. Pour l'entraînement, il s'agit de :
    
- a. supprimer tout ce qui est relatif d'un export vers GCS d'un package de code, 
- b. créer, si cela n'est pas déjà fait en amont, les données brutes d'entraînements au sein de la Data Registry depuis GCS, 
- c. créer et entraîner le modèle en lançant un CustomTrainingJob. Le modèle sera enristré dans la Model Registry alors que les
artefacts d'entraînement seront disponibles dans GCS.
   
- d. (optionnel) s'il est prévu de faire des prédictions à la volée, il faut créer un *Endpoint* et, dans un second temps,
y déployer le modèle. Le déploiement peut prendre plusieurs minutes.

En phase de prédiction, il s'agit, selon le type de prédiction voulu, de récupérer soit le endpoint, soit le modèle avant de récupérer la 
data et faire des prédictions.

Un exemple open-source peut être trouvé dans https://cloud.google.com/vertex-ai/docs/tutorials/tabular-bq-prediction/create-training-script?hl=fr .

# 3. Entraîner et prédire

## 3.A. Faire un entraînement avec Vertex AI
Nous voilà prêt pour passer dans le Cloud.

Pour accéder à la console Vertex AI, il suffit d’aller sur l’onglet des différentes fonctionnalités de la GCP et 
cliquer sur Vertex AI. Les différentes opérations peuvent y être suivies.

Pour lancer une opération d'entraînement, il suffit de lancer la même commande que pour le local, mais en remplaçant le contenu de 
l’argument d’environnement par la valeur nécessaire. C’est la seule modification à faire. 
```
python3 {CODE_PATH}/src/main.py --task train --config {PATH_TO_CONFIGURATION_FILE} --env cloud
```
# 3.B. Faire une prédiction avec Vertex AI
Dans le cadre de prédictions avec un Endpoint, la data **doit être** préprocessée en amont. Il n'est pas possible de le
faire en cours de prédiction.

Pour une prédiction avec l'Endpoint, il faut procéder aux étapes suivantes : 
1. Récupérer le endpoint ou le modèle selon le choix du type de prédiction.
2. Récupérer la data depuis GCS et passer d'un dataframe à une liste de listes.
3. Compléter le dataframe initial avec le résultat de la prédiction et renvoyer le résultat dans GCS.

# Nettoyage
Tout Endpoint non-supprimé est facturé. Ainsi, s'il n'y a pas d'intérêts à le garder, il faut le supprimer du Endpoint Registry.

# Pricing 
https://cloud.google.com/vertex-ai/pricing#europe