import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader,Dataset
from torchvision.transforms import ToTensor
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

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def train_model(dataset, num_epochs=10, batch_size=2, lr=0.005):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
    model = get_model(num_classes=2)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(num_epochs):
        model.train()
        for images, targets in dataloader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        lr_scheduler.step()

        print(f"Epoch: {epoch}, Loss: {losses.item()}")

    return model

if __name__ == "__main__":
    dataset = PascalVOCDataset(root_dir="VOCdataset", transform=ToTensor())
    trained_model = train_model(dataset)
    torch.save(trained_model.state_dict(), "head_segmentation_model.pth")
