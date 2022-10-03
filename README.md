# Diamond-Price-Prediction-Orchestrate-ML-Pipeline
# Managing Machine Learning Workflows using Prefect 2.0
# In this repository, you will find three versions of app
- version_1 - Breaking the Jupyter Notebook to Python Script (Basic Code without workflow management)
- version_2 - Code with Prefect Workflow - Defining the workflow and running them
- version_3 - Deployment and Scheduling tasks

# Why Prefect?
-Python based open source tool
-Manage ML Pipelines
-Schedule and Monitor the flow
-Gives observability into failures
-Native dask integration for scaling (Dask is used for parallel computing)
-Creating and activating a Virtual Environment
# In order to install prefect, create a virtual environment:
- $ python -m venv mlops

# Enter the Virtual Environment using below mentioned command:
- $ .\mlops\Scripts\activate

# Installing Prefect 2.0
Now install Prefect:
- $ pip install prefect
OR if you have Prefect 1, upgrade to Prefect 2 using this command:
- $ pip install -U prefect
OR to install a specific version:
- $ pip install prefect==2.4

# Check Prefect Version
Check the prefect version:
- $ prefect version

# Running Prefect Dashboard

![image](https://user-images.githubusercontent.com/65038531/193544685-564340f7-bcb6-4eec-b006-e4c9a1c3e4b0.png)

# Deployment of Prefect Flow
work_queue_name is used to submit the deployment to the a specific work queue.

You don't need to create a work queue before using the work queue. A work queue will be created if it doesn't exist.

from prefect.deployments import Deployment

from prefect.orion.schemas.schedules import IntervalSchedule

from datetime import timedelta

deployment = Deployment.build_from_flow(

    flow=main,
    
    name="model_training",
    
    schedule=IntervalSchedule(interval=timedelta(minutes=5)),
    
    work_queue_name="ml"
    
)

deployment.apply()

# Running an Agent
- $ prefect agent start --work-queue "ml"
