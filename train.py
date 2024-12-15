import mlflow
from ultralytics import YOLO
import os
import yaml
import matplotlib.pyplot as plt
from PIL import Image

from src.extract_info import process_dataset

if __name__ == '__main__':
    # Set MLflow experiment
    mlflow.set_tracking_uri("http://127.0.0.1:8081")
    mlflow.set_experiment("MLflow Quickstart")
    print(f"MLFlow URI: {mlflow.get_tracking_uri()}")

    # Paths
    dataset_yaml = 'dataset.yaml'
    output_folder = 'Dataset_Informations'
    print(f"Dataset YAML: {dataset_yaml}")
    print(f"Output folder: {output_folder}")

    # Start MLflow run 
    with mlflow.start_run():
        # Train YOLO model
        model = YOLO("yolov8n.pt")

        # Process dataset and generate visualizations
        process_dataset(dataset_yaml, output_folder)

        # Log artifacts (visualizations)
        mlflow.log_artifacts(output_folder, artifact_path=output_folder)
        print("Visualizations generated and saved.")
        results = model.train(data=dataset_yaml, epochs=3, imgsz=640)

        # Log parameters
        mlflow.log_param("dataset_yaml", dataset_yaml)

        # Log training parameters
        mlflow.log_param("epochs", 3)
        mlflow.log_param("imgsz", 640)

        # Add source tag
        mlflow.set_tag("mlflow.source.name", "train.py")