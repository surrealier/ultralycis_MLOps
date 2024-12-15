import mlflow
mlflow.set_tracking_uri(uri="http://127.0.0.1:8081")
mlflow.set_experiment("MLflow Quickstart")

from ultralytics import YOLO

import os
import yaml
import matplotlib.pyplot as plt
from PIL import Image
# import dvc

def extract_train_images(yaml_file_path, output_folder):
    # 1. YAML 파일에서 데이터 읽기
    with open(yaml_file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)

    # 2. train 경로 설정
    base_path = data['path']
    train_folder = os.path.join(base_path, data['train'])

    # 3. train_images 폴더 생성
    os.makedirs(output_folder, exist_ok=True)

    # 4. 최하위 폴더에서 이미지 추출
    for root, dirs, files in os.walk(train_folder):
        if not dirs:  # 최하위 폴더
            image_files = [f for f in files if f.lower().endswith(('png', 'jpg', 'jpeg'))]
            if image_files:
                first_image_path = os.path.join(root, image_files[0])

                # 이미지 열기
                image = Image.open(first_image_path)

                # 이미지 저장 경로 설정
                folder_name = os.path.basename(root)
                output_image_path = os.path.join(output_folder, f"{folder_name}.png")

                # 이미지 저장
                plt.imshow(image)
                plt.axis('off')
                plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
                plt.close()

if __name__ == '__main__':
    # YAML 파일 경로 및 출력 폴더 지정
    dataset_yaml = 'dataset.yaml'
    output_dir = 'train_images'

    # train 이미지 예시들 저장
    # extract_train_images(dataset_yaml, output_dir)

    with mlflow.start_run():
        model = YOLO("yolov8n.pt")
        mlflow.log_param("epochs", 3)
        mlflow.log_param("imgsz", 640)
        mlflow.log_param("data", "dataset.yaml")

        # 저장한 모든 이미지 artifact 저장
        mlflow.log_artifacts(output_dir, artifact_path="train_images")

        mlflow.set_tag("mlflow.source.name", "train.py")
        
        results = model.train(data=dataset_yaml, epochs=3, imgsz=640, workers=2)

        # mlflow.log_metric("mAP", results["mAP"])
        # mlflow.log_metric("P", results["P"])
        
        # mlflow.log_metric("R", results["R"])
        # mlflow.log_metric("F1", results["F1"])
        # mlflow.log_metric("val_loss", results["val_loss"])
        # mlflow.log_metric("train_loss", results["train_loss"])
        # mlflow.log_metric("test_loss", results["test_loss"])
        # mlflow.log_metric("test_mAP", results["test_mAP"])
        # mlflow.log_metric("test_P", results["test_P"])
        # mlflow.log_metric("test_R", results["test_R"])
        # mlflow.log_metric("test_F1", results["test_F1"])