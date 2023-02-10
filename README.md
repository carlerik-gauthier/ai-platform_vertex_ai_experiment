# Using AI-Platform

# Objectif

Le but est de fournir un template de code permettant de lancer un entraînement de modèle ML ainsi que des prédictions
avec la AI-Platform

# Data
Pour ce template, nous utilisons le jeu de données 'Titanic dataset' disponible sur Kaggle.

Il n'y a pas d'optimisation du modèle de prédiction de survie. Il est juste là pour illustrer.

# Steps

## 1. La data dans Storage, why ? 
## 2. Entraîner le modèle et sauver le modèle dans Storage depuis le local
## 3. Packaging
## 4. Faire une opération avec AI-Platform

# Pré-requis

**A.** Avoir les credentials GCP; en particulier pour Storage. 

**B.** Dans les arguments du main.py, définir une variable qui permet de savoir s'il faut lancer les calculs avec la 
AI-Platform ou le faire en local.

Ici, cet argument s'appelle env.

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

### B. L'exécutable lancée par la AI-Platform
Il s'agit du fichier qui sera lancée par la AI-Platform. Il est le point d'entrée pour les opérations de calculs à faire 
dans le Cloud. Lors de calculs en local, le code principal importe les fonctions nécessaires se trouvant dans ce module.

Le but de ce point est de s'assurer que la récupération des arguments et que les bonnes informations sont transmises aux 
fonctions de calculs.

Ici, il s'agit du fichier src/models/classifier.py

**NB:** json.dumps transforme un dictionnaire en string : json.loads({dict}) = ‘{dict}’; 
alors que json.loads procède à l'opération inverse : json.loads('{dict}') = {dict}.            

# 3. Packaging
1. Il s'agit de créer un fichier de setup, nommé setup.py, qui devra se trouver au même niveau d'arborescence que les 
modules à exécuter dans le Cloud. Pour de plus amples informations, voir

https://cloud.google.com/ai-platform/training/docs/packaging-trainer 

Toutefois, quand un fichier de requirements est prévu, il est préférable de créer une distribution wheel 
2. Après s'être positionné au bon niveau de l'arborescence, exécuter la commande 
```
python setup.py bdist_wheel
```
Cela créera 3 folders au même niveau que le fichier de setup : {setup_name}.egg-info, build et dist. 
Dans ce dernier folder se trouve le package sous le nom {setup_name}-{version}-py3-none-any.whl .

3. déplacer la distribution wheel de src/dist vers le folder package. Le code va la chercher  à cet endroit
avant de la déplacer dans Storage où la AI-Platform va la récupérer.

**NB** : Si entre 2 runs dans le Cloud, il y a une modification de codes, il faut suivre à nouveau la procédure de 
packaging. Sinon, les changements ne seront pas pris en compte.
# 4. Faire une opération avec AI-Platform
Nous voilà prêt pour passer dans le Cloud.

Pour accéder à la console AI-Platform, il suffit d’aller sur l’onglet des différentes fonctionnalités de la GCP et 
cliquer sur AI-Platform. 

Pour lancer une opération, il suffit de lancer la même commande que pour le local, mais en remplaçant le contenu de 
l’argument d’environnement par la valeur nécessaire. C’est la seule modification à faire. 
```
python3 {CODE_PATH}/src/main.py --task {TASK} --config {PATH_TO_CONFIGURATION_FILE} --env cloud
```

**NB** : Les premiers essais permettent de détecter les éventuelles erreurs avant même un lancement effectif sur la 
AI-Platform.

