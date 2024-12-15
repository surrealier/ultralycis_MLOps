import os
import yaml
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import numpy as np
from collections import defaultdict
import matplotlib.patches as patches
import networkx as nx
import pygraphviz
import pydot

def load_yaml(yaml_file_path):
    """Load and parse YAML file."""
    with open(yaml_file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def get_representative_image_paths(base_folder):
    """Get one representative image and its label file path per lowest-level folder."""
    image_paths = []
    label_paths = []
    for root, dirs, files in os.walk(base_folder):
        if not dirs:  # Lowest-level folder
            images = [f for f in files if f.lower().endswith(('png', 'jpg', 'jpeg'))]
            labels = [f for f in files if f.lower().endswith('txt')]
            if images and labels:
                image_paths.append(os.path.join(root, images[0]))
                label_paths.append(os.path.join(root, labels[0]))
    return image_paths, label_paths

def get_all_image_and_label_paths(base_folder):
    """Get all image and label file paths from the given folder."""
    image_paths, label_paths = [], []
    for root, _, files in os.walk(base_folder):
        image_paths.extend(
            os.path.join(root, f) for f in files if f.lower().endswith(('png', 'jpg', 'jpeg'))
        )
        label_paths.extend(
            os.path.join(root, f) for f in files if f.lower().endswith('txt')
        )
    return image_paths, label_paths

def save_image_with_bbox(image_path, label_path, class_names, output_folder):
    """Draw bounding boxes on the image and save the result."""
    image = Image.open(image_path)
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    with open(label_path, 'r') as lf:
        for line in lf.readlines():
            cls_id, x_center, y_center, width, height = map(float, line.strip().split())
            cls_name = class_names[int(cls_id)]
            if cls_id == 0:
                color = 'white'
            elif cls_id == 1:
                color = 'blue'
            elif cls_id == 2:
                color = 'green'
            elif cls_id == 3:
                color = 'yellow'
            elif cls_id == 4:
                color = 'orange'
            elif cls_id == 5:
                color = 'purple'
            elif cls_id == 6:
                color = 'brown'
            elif cls_id == 7:
                color = 'pink'
            elif cls_id == 8:
                color = 'gray'
            elif cls_id == 9:
                color = 'cyan'
            else:
                color = 'red'
            x, y = x_center - width / 2, y_center - height / 2
            
            rect = patches.Rectangle(
                (x * image.width, y * image.height),
                width * image.width,
                height * image.height,
                linewidth=1,
                edgecolor=color,
                facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(
                x * image.width,
                max(y * image.height - 10, 0),
                cls_name,
                color=color,
                fontsize=8,
                bbox=dict(facecolor=color, alpha=0.5, edgecolor='none')
            )

    os.makedirs(output_folder, exist_ok=True)
    output_image_path = os.path.join(output_folder, os.path.basename(image_path))
    plt.axis('off')
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def plot_class_distribution(base_folder, class_names, output_file):
    """Plot and save class distribution."""
    _, label_paths = get_all_image_and_label_paths(base_folder)
    class_counts = {cls: 0 for cls in class_names}

    for label_path in label_paths:
        with open(label_path, 'r') as lf:
            for line in lf.readlines():
                cls_id = int(line.strip().split()[0])
                class_counts[cls_id] += 1

    sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()))
    plt.xticks(rotation=45)
    plt.title("Class Distribution")
    plt.savefig(output_file)
    plt.close()

def plot_image_size_distribution(base_folder, output_file):
    """Plot and save image size distribution."""
    image_paths, _ = get_all_image_and_label_paths(base_folder)
    widths, heights = [], []

    for image_path in image_paths:
        with Image.open(image_path) as img:
            widths.append(img.width)
            heights.append(img.height)

    plt.hist(widths, bins=30, alpha=0.7, label='Widths')
    plt.hist(heights, bins=30, alpha=0.7, label='Heights')
    plt.legend()
    plt.title("Image Size Distribution")
    plt.savefig(output_file)
    plt.close()

def plot_bbox_ratio_distribution(bbox_ratios, output_file):
    """Plot and save bounding box size ratio distribution by class."""
    plt.figure(figsize=(12, 8))
    for cls_name, ratios in bbox_ratios.items():
        if ratios:
            plt.hist(ratios, bins=50, alpha=0.6, label=cls_name)

    plt.title("Bounding Box Size Ratio Distribution by Class")
    plt.xlabel("Bounding Box Size Ratio (Width * Height)")
    plt.ylabel("Frequency")
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def count_files_and_subfiles(folder_path):
    """폴더 및 하위 폴더에 있는 모든 파일 개수를 합산."""
    total_files = 0
    for root, _, files in os.walk(folder_path):
        total_files += len(files)
    return total_files

def build_folder_graph(base_path, max_display=2, depth=0):
    """폴더 구조 그래프 생성"""
    G = nx.DiGraph()
    
    def add_nodes_edges(current_path, parent_node, depth):
        subfolders = sorted([
            f for f in os.listdir(current_path) if os.path.isdir(os.path.join(current_path, f))
        ])
        total_files = count_files_and_subfiles(current_path)

        # 노드 추가
        folder_name = os.path.basename(current_path)
        label = f"{folder_name}\n({total_files} files)"
        G.add_node(current_path, label=label)
        if parent_node:
            G.add_edge(parent_node, current_path)

        # 하위 폴더 처리
        if len(subfolders) > max_display and depth > 0:
            import pdb; pdb.set_trace()
            # 최하위 폴더인지 확인
            if not os.listdir(current_path):
                for subfolder in subfolders[:max_display]:
                    subfolder_path = os.path.join(current_path, subfolder)
                    add_nodes_edges(subfolder_path, current_path, depth)

                # 생략 표시
                ellipsis_node = f"{current_path}_ellipsis"
                G.add_node(ellipsis_node, label="...")
                G.add_edge(current_path, ellipsis_node)
        else:
            for subfolder in subfolders:
                print(f"subfolder: {subfolder}, depth: {depth}")
                subfolder_path = os.path.join(current_path, subfolder)
                add_nodes_edges(subfolder_path, current_path, depth + 1)

    
    # change \\ to / for windows
    base_path = base_path.replace("\\", "/")
    add_nodes_edges(base_path, None, depth)
    return G

def plot_folder_graph(G, output_path):
    """폴더 구조 그래프 시각화 및 저장장"""
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot", args="-Grankdir=LR")
    labels = nx.get_node_attributes(G, 'label')
    plt.figure(figsize=(19, 14))
    nx.draw(
        G, pos, with_labels=True, labels=labels, node_size=3000, 
        node_color="skyblue", font_size=10, font_weight="bold", edge_color="gray"
    )
    plt.title("Folder Structure")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def process_dataset(yaml_file_path, output_folder):
    """Main function to process the dataset and generate visualizations."""
    data = load_yaml(yaml_file_path)
    base_path = data['path']
    class_names = data['names']

    train_folder = os.path.join(base_path, data['train'])
    val_folder = os.path.join(base_path, data['val'])

    # Create separate output folders for train and val
    train_output_folder = os.path.join(output_folder, 'train')
    val_output_folder = os.path.join(output_folder, 'val')

    # Visualize representative bounding boxes and save
    for folder, sub_output_folder in [(train_folder, train_output_folder), (val_folder, val_output_folder)]:
        sub_output_images_folder = os.path.join(sub_output_folder, 'images')
        image_paths, label_paths = get_representative_image_paths(folder)
        bbox_ratios = defaultdict(list)

        for image_path, label_path in zip(image_paths, label_paths):
            save_image_with_bbox(image_path, label_path, class_names, sub_output_images_folder)

            with open(label_path, 'r') as lf:
                for line in lf.readlines():
                    _, _, _, width, height = map(float, line.strip().split())
                    bbox_ratios[class_names[int(line.split()[0])]].append(width * height)

        plot_bbox_ratio_distribution(
            bbox_ratios, os.path.join(sub_output_folder, "bbox_ratio_distribution.png")
        )

    # Plot class and image size distributions for train and val
    plot_class_distribution(train_folder, class_names, os.path.join(train_output_folder, "class_distribution.png"))
    plot_class_distribution(val_folder, class_names, os.path.join(val_output_folder, "class_distribution.png"))
    print("#1 - plot_class_distribution DONE")
    plot_image_size_distribution(train_folder, os.path.join(train_output_folder, "image_size_distribution.png"))
    plot_image_size_distribution(val_folder, os.path.join(val_output_folder, "image_size_distribution.png"))
    print("#2 - plot_image_size_distribution DONE")
    train_folder_graph = build_folder_graph(train_folder)
    val_folder_graph = build_folder_graph(val_folder)
    plot_folder_graph(train_folder_graph, os.path.join(train_output_folder, "train_folder_structure.png"))
    plot_folder_graph(val_folder_graph, os.path.join(val_output_folder, "val_folder_structure.png"))
    print("#3 - visualize_folder_structure DONE")