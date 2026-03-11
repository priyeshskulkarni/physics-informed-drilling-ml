# Physics-Informed Drilling ML

Physics-Informed Machine Learning framework for real-time prediction of drilling fluid rheology using sensor data.

## Features

- Synthetic drilling dataset generator
- Physics-informed neural network (PINN)
- Training pipeline using PyTorch
- FastAPI inference server
- Real-time prediction endpoint
- Docker-ready architecture

## Run the Project

Generate dataset:

python data/synthetic_generator.py

Train model:

python -m training.train_pinn

Run API:

uvicorn inference.api_server:app --reload
