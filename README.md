# ai-platform_vertex_ai_experiment
The purpose of this repo is to regroup code I used when using GCP AI-Platform and its new product Vertex AI

I've already used AI-Platform wrt production condition.

Concerning Vertex AI, I aim to test different ways to use it and see differences with the use of AI-Platform

The 'main' branch remains empty and every other option is displayed in another branch.

As of February 14, 2023, the branches are :

- local : the core code one would write when developing its modules
- ai-platform : shows the adaptations to be done in order to perform computations with GCP AI-Platform
- vertex-ai-1-prebuilt-images : shows how the code should be modified in order to use Vertex AI using CustomJob and 
prebuilt images for training a model and using it with Endpoints.
- vertex-ai-2-custom-python-package : shows how to modify the code in order to use Vertex AI using 
CustomPythonPackageJob and prebuilt images for training a model and using it with Endpoints
- vertex-ai-3-custom-images : shows how the code can be adapted in order to use Custom Images with Vertex AI
- vertex-ai-4-cpr : this branch aims to give a way to implement Custom Prediction Routines with Vertex AI. Images are
pre-built

## Status
Done
