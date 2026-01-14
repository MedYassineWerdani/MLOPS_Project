from zenml import pipeline, step
from model_utils import train_model, TransformerClassifier
import torch.nn as nn
import os

@step
def prepare_data() -> str:
    data_path = "data/data.tsv"
    if not os.path.exists(data_path):
        raise FileNotFoundError("Data file missing!")
    return data_path

@step(enable_cache=True)
def training_step(data_path: str) -> nn.Module:
    """Trains the model and returns it for ZenML to version."""
    model = train_model(data_path, epochs=1) 
    return model

@pipeline
def political_bias_pipeline():
    path = prepare_data()
    training_step(path)

if __name__ == "__main__":
    # Just call the pipeline function
    political_bias_pipeline()