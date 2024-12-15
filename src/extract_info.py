import os
import yaml
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import numpy as np
from collections import defaultdict
import matplotlib.patches as patches

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

def visualize_folder_structure(base_folder, output_image_path):
    """Visualize the folder structure and file counts."""
    folder_dict = defaultdict(lambda: defaultdict(int))

    for root, dirs, files in os.walk(base_folder):
        relative_path = os.path.relpath(root, base_folder)
        parts = relative_path.split(os.sep)
        if len(parts) > 1:
            parent_folder = os.sep.join(parts[:-1])
            folder_dict[parent_folder][parts[-1]] += len(files)
        else:
            folder_dict[relative_path]['_files'] += len(files)

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = 0
    for parent, subfolders in folder_dict.items():
        subfolder_names = list(subfolders.keys())
        if len(subfolder_names) > 2:
            subfolder_names = subfolder_names[:2] + [f"... {len(subfolder_names) - 2} more"]
        for subfolder in subfolder_names:
            ax.text(0, y_pos, f"{parent}/{subfolder}", fontsize=12)
            if subfolder != '_files':
                ax.text(1, y_pos, f"{subfolders[subfolder]} files", fontsize=12)
            y_pos -= 1

    ax.set_xlim(0, 2)
    ax.set_ylim(y_pos, 1)
    ax.axis('off')
    plt.savefig(output_image_path, bbox_inches='tight')
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
            bbox_ratios, os.path.join(sub_output_images_folder, "bbox_ratio_distribution.png")
        )

    # Plot class and image size distributions for train and val
    plot_class_distribution(train_folder, class_names, os.path.join(train_output_folder, "class_distribution.png"))
    plot_class_distribution(val_folder, class_names, os.path.join(val_output_folder, "class_distribution.png"))
    print("#1 - plot_class_distribution DONE")
    plot_image_size_distribution(train_folder, os.path.join(train_output_folder, "image_size_distribution.png"))
    plot_image_size_distribution(val_folder, os.path.join(val_output_folder, "image_size_distribution.png"))
    print("#2 - plot_image_size_distribution DONE")
    visualize_folder_structure('train', os.path.join(train_output_folder, "train_folder_structure.png")) 
    visualize_folder_structure('val', os.path.join(val_output_folder, "val_folder_structure.png"))
    print("#3 - visualize_folder_structure DONE")