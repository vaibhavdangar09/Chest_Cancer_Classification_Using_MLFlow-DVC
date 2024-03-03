# End_To_End_Animal_Classification_Using_MLFlow-DVC

# Workflow

1. Update config.yaml

2. Update secrets.yaml [Optional]

3. Update params.yaml

4. Update the entity

5. Update the configuration manager in src config

6. Update the components

7. Update the pipeline

8. Update the main.py

9. Update the dvc.yaml


# MLFLOW
[Documentation](https://mlflow.org/docs/latest/index.html)

cmd
* mlflow ui

# dagshub
[dagshub](https://dagshub.com/dashboard)

MLFLOW_TRACKING_URI=https:Your MLFlow Uri
MLFLOW_TRACKING_USERNAME=Your Github Username
MLFLOW_TRACKING_PASSWORD=Your MLFlow Tracking Password
python script.py

Run this to export as env variables:

```bash
 
export MLFLOW_TRACKING_URI=Your MLFlow Uri

export MLFLOW_TRACKING_USERNAME=Your Github username

export MLFLOW_TRACKING_PASSWORD=Your MLFlow Tracking Password

```

# DVC cmd
 1. dvc init
 2. dvc repro
 3. dvc dag

# About MLflow & DVC
MLflow

* Its Production Grade
* Trace all of your expriements
* Logging & taging your model

DVC

* Its very lite weight for POC only
* lite weight expriements tracker
* It can perform Orchestration (Creating Pipelines)

