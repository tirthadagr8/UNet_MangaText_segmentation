import xml.etree.ElementTree as ET
from PIL import Image
import os
from torch.utils.data import DataLoader, Dataset
from time import sleep
from tqdm import tqdm
import pandas as pd
import unicodedata

def convert_fullwidth_to_halfwidth(text):
    normalized_text = unicodedata.normalize('NFKC', text)
    return ''.join(c for c in unicodedata.normalize('NFD', normalized_text) if unicodedata.category(c) != 'Mn')

# Helper function to convert bounding box to x1, y1, x2, y2, x3, y3, x4, y4
def bbox_to_coordinates(xmin, ymin, xmax, ymax):
    return [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]

def load_data():
    xml_dir = 'C:/Users/tirth/Downloads/Manga109/Manga109_released_2023_12_07/annotations/'  # Directory containing XML files
    image_dir = 'C:/Users/tirth/Downloads/Manga109/Manga109_released_2023_12_07/images/'  # Directory containing page images

    images=[]
    annotations = []
    labels=[]

    # Iterate over all XML files
    for xml_file in tqdm(os.listdir(xml_dir)):
        if not xml_file.endswith('.xml'):
            continue
        # print(xml_file)
        # sleep(5)
        # Parse the XML file
        tree = ET.parse(os.path.join(xml_dir, xml_file))
        root = tree.getroot()
        
        # Process each page in the XML
        for page in root.find('pages'):
            page_index = page.get('index')
            
            # Load the corresponding image for the page
            page_image_path = os.path.join(image_dir, f'{xml_file[:-4]}/{str(page_index).zfill(3)}.jpg')
            if not os.path.exists(page_image_path):
                print(f"Warning: Image for page {page_index} not found.")
                continue

            images.append(page_image_path)
            
            # Prepare data for the first model (page-level data with annotations)
            coo=[]
            texts=''
            for text in page.findall('text'):
                # Extract text annotation coordinates
                xmin, ymin = int(text.get('xmin')), int(text.get('ymin'))
                xmax, ymax = int(text.get('xmax')), int(text.get('ymax'))
                coordinates = (xmin, ymin, xmax, ymin, xmax,ymax,xmin, ymax) #bbox_to_coordinates(xmin, ymin, xmax, ymax)
                
                coo.append(coordinates)
                texts+=(convert_fullwidth_to_halfwidth(text.text.strip()) if text.text else "")
            annotations.append(coo)
            labels.append(texts)
    return images, annotations, labels

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from transformers import AutoTokenizer

class DatasetLoaderPrivate2(Dataset):
    def __init__(self, images, labels, annotations, config) -> None:
        super().__init__()
        self.config=config
        self.max_length = 100
        self.img_size = config.img_size
        self.transforms = T.Compose([
            T.Resize((self.img_size, self.img_size)),
            T.RandomPerspective(distortion_scale=0.2, p=0.5),
            T.RandomRotation(5),
            T.GaussianBlur(3),
            T.RandomAdjustSharpness(2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.simple_transforms = T.Compose([
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.tokenizer = AutoTokenizer.from_pretrained('tirthadagr8/CustomOCR')
        self.images = images
        self.labels = labels
        self.annotations = annotations

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        augment = False
        if index >= len(self.images):
            index %= len(self.images)
            augment = True
        if len(self.annotations[index])==0:
            index=100
        img = Image.open(self.images[index]).convert('RGB')
        original_width, original_height = img.size

        # Apply transformations
        if augment:
            img = self.transforms(img)
        else:
            img = self.simple_transforms(img)

        _, new_height, new_width = img.shape  # Shape: (C, H, W)
        scale_x = new_width / original_width
        scale_y = new_height / original_height

        # Convert annotations to bounding boxes
        bbox = self.convert_to_bbox(np.array(self.annotations[index]), scale_x, scale_y, self.max_length)
        labels = torch.zeros((len(bbox),), dtype=torch.int64)

        # Generate masks for the bounding boxes
        mask = self.generate_mask(bbox, new_height, new_width)

        return {
            'pixel_values': img,
            'boxes': bbox,
            'masks': mask,
            'class_labels': labels,
        }

    def convert_to_bbox(self, vertices, scale_x=1, scale_y=1, max_length=512):
        """
        Converts an array of 8-point vertices to 4-point bounding boxes.
        
        Args:
            vertices (np.ndarray): A 2D array of shape (num_boxes, 8) containing 8 integers per bounding box.
        
        Returns:
            torch.Tensor: A tensor of shape (num_boxes, 4) containing 4 integers per bounding box.
        """
        # Extract xmin, ymin, xmax, ymax
        xmin = vertices[:, 0]  # First column
        ymin = vertices[:, 1]  # Second column
        xmax = vertices[:, 4]  # Fifth column
        ymax = vertices[:, 5]  # Sixth column
        
        # Stack the values into a single array
        bbox = np.stack([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0, xmax - xmin, ymax - ymin], axis=1)
        scaled_boxes = []
        for box in bbox:
            x1, y1, x2, y2 = box
            # Scale coordinates
            x1 = x1 * scale_x
            y1 = y1 * scale_y
            x2 = x2 * scale_x
            y2 = y2 * scale_y
            scaled_boxes.append([x1, y1, x2, y2])

        # Calculate padding size
        num_boxes = bbox.shape[0]
        padding_size = max_length - num_boxes
        # padding_size=0
        if padding_size > 0:
            # Create padding array with [0, 0, 0, 0]
            padding = np.zeros((padding_size, 4), dtype=np.float32)
            scaled_boxes = np.vstack([scaled_boxes, padding])  # Append padding to the bounding boxes

        # Convert to PyTorch tensor
        return torch.tensor(scaled_boxes, dtype=torch.float32) / self.img_size

    def generate_mask(self, bbox, height, width):
        """
        Generates a single binary mask for all bounding boxes.
        
        Args:
            bbox (torch.Tensor): A tensor of shape (num_boxes, 4) containing normalized bounding boxes.
            height (int): Height of the image.
            width (int): Width of the image.
        
        Returns:
            torch.Tensor: A binary mask tensor of shape (1, height, width).
        """
        # Create a single mask for all bounding boxes
        mask = torch.zeros((1, height, width), dtype=torch.float32)  # Single channel mask
    
        for i, box in enumerate(bbox):
            # Unnormalize the bounding box coordinates
            cx, cy, w, h = box * self.img_size
            x_min = int(cx - w / 2)
            y_min = int(cy - h / 2)
            x_max = int(cx + w / 2)
            y_max = int(cy + h / 2)
    
            # Clamp the coordinates to ensure they are within the image bounds
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(width - 1, x_max)
            y_max = min(height - 1, y_max)
    
            # Set the mask region to 1
            mask[0, y_min:y_max + 1, x_min:x_max + 1] = 1.0
    
        return mask

if __name__ == "__main__":
    from config import Config
    from utility import *
    images, annotations, labels = load_data()
    print(f"Number of images: {len(images)}")
    print(f"Number of annotations: {len(annotations)}")
    print(f"Number of labels: {len(labels)}")
    print(images[0], annotations[0], labels[0])
    config=Config()
    dataset = DatasetLoaderPrivate2(images, labels, annotations, config)
    sample = dataset[0]
    print(sample['pixel_values'].shape)
    print(sample['boxes'].shape)
    print(sample['masks'].shape)
    print(sample['class_labels'].shape)
    d=DataLoader(dataset,batch_size=2,shuffle=True)
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    c=0
    ttt=[]
    fig,ax=plt.subplots()
    for t in tqdm(d):
        _im=t['pixel_values'][0].detach().cpu().permute(1,2,0).numpy()
        plt.imshow(_im)
        for box in t['boxes'][0]:
            box*=config.img_size
            rect = patches.Rectangle((box[0]-box[2]/2.0,box[1]-box[3]/2.0), box[2].item(), box[3].item(), linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        plt.show()

        c+=1
        break