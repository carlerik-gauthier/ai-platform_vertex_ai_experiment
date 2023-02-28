# Branche Vertex AI avec des images CUSTOM

# Documentation

- https://cloud.google.com/docs?hl=fr
- https://cloud.google.com/vertex-ai/docs/training/create-training-pipeline#custom-container
- https://cloud.google.com/artifact-registry/docs/repositories/create-repos?hl=fr#gcloud
- https://cloud.google.com/vertex-ai/docs/predictions/pre-built-containers?hl=fr
- https://cloud.google.com/vertex-ai/docs/training/pre-built-containers
- https://cloud.google.com/vertex-ai/docs/training/create-custom-container
- https://cloud.google.com/vertex-ai/docs/training/custom-training
- google-cloud-aiplatform : https://github.com/googleapis/python-aiplatform/
- https://cloud.google.com/vertex-ai/docs/predictions/use-custom-container#aiplatform_upload_model_highlight_container-python (PREDICT)
- https://www.youtube.com/watch?v=VRQXIiNLdAk&t=516s (TRAIN) 
- https://www.youtube.com/watch?v=EOBJYnavwfw (TRAIN) 
- https://www.youtube.com/watch?v=-9fU1xwBQYU&list=PLIivdWyY5sqJAyUJbbsc8ZyGLNT4isnuB&index=5 (PREDICT)
- https://codelabs.developers.google.com/vertex-p2p-training#5
- https://codelabs.developers.google.com/vertex-p2p-predictions?hl=en#0
- https://medium.com/google-cloud/how-to-train-ml-models-with-vertex-ai-training-f9046bfbcfab 
- https://cloud.google.com/python/docs/reference/aiplatform/latest#google.cloud.aiplatform.CustomContainerTrainingJob
- https://github.com/GoogleCloudPlatform/mlops-on-gcp/blob/master/on_demand/kfp-caip-sklearn/lab-01-caip-containers/lab-01.ipynb
- https://stackoverflow.com/questions/35153902/find-the-list-of-google-container-registry-public-images
- https://blog.searce.com/deploy-your-own-custom-ml-on-vertex-ai-using-gcp-console-e3c52f7da2b
- https://blog.ml6.eu/vertex-ai-is-all-you-need-599ffc9473fd
- https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/migration/UJ3%20Vertex%20SDK%20Custom%20Image%20Classification%20with%20custom%20training%20container.ipynb
- https://blog.ml6.eu/deploy-ml-models-on-vertex-ai-using-custom-containers-c00f57efdc3c
- https://adswerve.com/blog/how-to-build-a-customized-vertex-ai-container/
- https://cloud.google.com/deep-learning-containers/docs/choosing-container?hl=fr
- https://sourabhsjain.medium.com/model-training-using-google-cloud-ai-platform-custom-containers-ca7348fbfb81
- https://cloud.google.com/vertex-ai/docs/training/code-requirements?hl=fr
- https://cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements

- https://github.com/GoogleCloudPlatform/training-data-analyst/tree/master/self-paced-labs/vertex-ai/vertex-ai-qwikstart

Pipelines
- https://cloud.google.com/blog/topics/developers-practitioners/using-vertex-ai-rapid-model-prototyping-and-deployment?hl=en 
- https://codelabs.developers.google.com/vertex-pipelines-intro#0
- https://towardsdatascience.com/how-to-set-up-custom-vertex-ai-pipelines-step-by-step-467487f81cad

Other
- https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/bigquery_ml/bqml-online-prediction.ipynb
- https://cloud.google.com/vertex-ai/docs/training/containerize-run-code-local?hl=fr
- https://cloud.google.com/artifact-registry/docs/repositories/create-repos?hl=fr#gcloud
- https://cloud.google.com/vertex-ai/docs/training/using-managed-datasets?hl=fr
- https://console.cloud.google.com/gcr/images/google-containers/GLOBAL
- https://console.cloud.google.com/gcr/images/kubeflow-images-public/GLOBAL
- https://console.cloud.google.com/gcr/images/deeplearning-platform-release/GLOBAL
- https://codelabs.developers.google.com/?text=vertex
- https://cloud.google.com/vertex-ai/docs/reference/rest/v1/CustomJobSpec#FIELDS.base_output_directory


# Objectif

Le but est de fournir un template de code permettant de lancer un entraînement de modèle ML avec Vertex AI en utilisant
des images customs. Il s'agit de l'approche qui permet d'avoir le plus de latitude. Toutefois, elle est plus compliquée.

La documentation Vertex AI recommande cette approche uniquement si aucune image "pré-construite" ne permet de contenir
le code du workflow développé.

# Data
Pour ce template, nous utilisons le jeu de données 'Titanic dataset' disponible sur Kaggle.

Il n'y a pas d'optimisation du modèle de prédiction de survie. Il est juste là pour illustrer.

# Steps
### 1. Ce qui change avec le cas des images pre-built 
### 2. Créer et publier l'image Docker d'entraînement
### 3. Entraîner et prédire

# Pré-requis
**0.** Activer les APIs Vertex AI et Artifact Registry

**A.** Avoir les credentials GCP; en particulier pour Storage.  

**B.** Dans les arguments du main.py, définir une variable qui permet de savoir s'il faut lancer les calculs avec 
Vertex AI ou le faire en local. Par ailleurs, il existe une notion de version pour l'endpoint

Ici, ces arguments s'appellent env et endpoint_version.

# 1. Ce qui change avec le cas des images pre-built 
Cette approche permet de conserver un découpage des tâches de préprocessing et de création/entraînement du modèle en 
2 fichiers distincts tout comme cela est fait pour un job AI-Platform ou au moyen d'un "CustomPythonPackage" *sans* avoir
à faire un packaging explicite. Cela est possible par la création d'une image Docker et sa publication au sein de la 
"Artifact Registry".

# 2. Créer et publier l'image Docker d'entraînement      
Les spécifications générales sont disponible à ce lien https://cloud.google.com/vertex-ai/docs/training/create-custom-container .

La structure élémentaire du Dockerfile est la suivante
```bash
FROM "<base image>" # it can be an image available in the public Artifact Registry. In that case, it has the form gcr.io/.../...
# https://console.cloud.google.com/gcr/images/deeplearning-platform-release/GLOBAL

WORKDIR "<code's work directory>"

COPY "<code local directory>" "<directory in the image>"

RUN pip install -r <requirements file>

ENTRYPOINT ["python", "-m", "<relative path to module from Docker root>"]
```
Pour construire et publier l'image sur Artifact Registry, il suffit de suivre les instructions décrites dans
**artifact/push_img_to_artifact.sh**

Le Dockerfile doit se trouver au même niveau (à priori) que src :
```bash
.
├── Dockerfile
├── README.md
├── src
└── venv
```

# 3.A. Faire un entraînement 
Comme pour le cas des images 'pre-built'.

# 3.B. Faire une prédiction 
Comme pour le cas des images 'pre-built'.

Tout comme la phase d'entraînement, il est possible de créer une image custom pour la phase prédictive. Le nom de cette 
image est à fournir à la variable **model_serving_container_image_uri** du job *aiplatform.CustomContainerTrainingJob*.
Toutefois, cela n'est pas montré dans ce repo. J'invite donc le lecteur à voir les références suivantes pour voir les 
prérequis et s'inspirer de la façon de procéder :
- https://cloud.google.com/vertex-ai/docs/predictions/use-custom-container#aiplatform_upload_model_highlight_container-python
- https://www.youtube.com/watch?v=-9fU1xwBQYU&list=PLIivdWyY5sqJAyUJbbsc8ZyGLNT4isnuB&index=5
- https://blog.ml6.eu/vertex-ai-is-all-you-need-599ffc9473fd
- https://blog.ml6.eu/deploy-ml-models-on-vertex-ai-using-custom-containers-c00f57efdc3c
- https://adswerve.com/blog/how-to-build-a-customized-vertex-ai-container/
# Pricing 
https://cloud.google.com/vertex-ai/pricing#europe