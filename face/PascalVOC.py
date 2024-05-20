import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET

class PascalVOCDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_dir = os.path.join(root_dir, "JPEGImages")
        self.annotation_dir = os.path.join(root_dir, "Annotations")
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        annot_path = os.path.join(self.annotation_dir, self.image_files[idx].replace('.jpg', '.xml'))
        boxes, labels = self.parse_voc_xml(annot_path)

        if self.transform:
            image = self.transform(image)

        return image, {"boxes": boxes, "labels": labels}

    def parse_voc_xml(self, annot_path):
        tree = ET.parse(annot_path)
        root = tree.getroot()
        boxes = []
        labels = []
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1)  # Assuming 'head' class label is 1
        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)
