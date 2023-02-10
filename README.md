# Vertex AI avec des images pre-built ET des CustomPythonPackageTrainingJob

# Links
- https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform
- https://cloud.google.com/artifact-registry/docs/python/manage-packages#gcloud
- https://cloud.google.com/artifact-registry/docs/repositories/create-repos?hl=fr#gcloud
- https://cloud.google.com/vertex-ai/docs/training/create-training-pipeline#pre-built-container 
- https://cloud.google.com/vertex-ai/docs/training/pre-built-containers
- https://cloud.google.com/vertex-ai/docs/predictions/pre-built-containers?hl=fr
- https://github.com/googleapis/python-aiplatform/ 
# Objectif

Le but est de fournir un template de code permettant de lancer un entraînement de modèle ML avec Vertex AI en utilisant
des images pre-built par une approche "package" telle qu'utilisée pour un job AI-Platform.

# Data
Pour ce template, nous utilisons le jeu de données 'Titanic dataset' disponible sur Kaggle.

Il n'y a pas d'optimisation du modèle de prédiction de survie. Il est juste là pour illustrer.

# Steps
## 1. Packaging
## 2. Ce qui change par rapport au cas des images pre-built (branche vertex-ai) et AI-Platform
## 3. Entraîner et prédire

# Pré-requis
**0.** Activer les APIs Vertex AI et Artifact Registry

**A.** Avoir les credentials GCP; en particulier pour Storage.  

**B.** Dans les arguments du main.py, définir une variable qui permet de savoir s'il faut lancer les calculs avec 
Vertex AI ou le faire en local. Par ailleurs, il existe une notion de version pour l'endpoint

Ici, ces arguments s'appellent env et endpoint_version.

# 1. Packaging
Le packaging se fait comme pour le cas AI-Platform.

# 2. Ce qui change par rapport au cas des images pre-built (branche vertex-ai) et AI-Platform
Cette approche est la plus similaire de ce qui est fait pour un job AI-Platform. D'un autre côté, le découpage du main.py
est le même que pour la branche "vertex-ai". 

La différence notable est qu'au lieu d'appeler un script 'auto-suffisant', il faut fournir le package (en précisant l'URI 
GCS) et le module python servant de point d'entrée.

# 3.A. Faire un entraînement avec Vertex AI
Comme pour le cas "pre-built" images
# 4.B. Faire une prédiction avec Vertex AI
Comme pour le cas "pre-built" images

# Pricing 
https://cloud.google.com/vertex-ai/pricing#europe

# Miscellaneous 
Il est possible de pousser dans la Artifact Registry un artifact de format Python. 
Au final et bien que cela ne s'avère pas utile pour l'utilisation d'un entraînement avec un "custom python package job",
il peut être utile de connaître les étapes. Elles sont :

### Installation
pip install --upgrade pip setuptools wheel
pip install twine keyrings.google-artifactregistry-auth
gcloud services enable artifactregistry.googleapis.com

### create a python repository if wanted
gcloud artifacts repositories create <your_repository_name> --repository-format=python 
--location=<your_repository_location>
--description="<your_repository_description>"

### build the package
python setup.py bdist_wheel

cp dist/{package_name}-{package_version}-py3-none-any.whl ../package/

### Deploy the sample python package to artifact registry (dist has to be a subfolder from current position)
twine upload --repository-url https://<your_repository_location>-python.pkg.dev/<your_gcp_project>/<your_repository_name>/ dist/*