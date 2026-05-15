import torch
import os
import pandas as pd
from PIL import Image

class YOLODataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir, self.label_dir = img_dir, label_dir
        self.transform = transform
        self.S, self.B, self.C = S, B, C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for line in f.readlines():
                boxes.append([float(x) for x in line.replace("\n", "").split()])

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        
        if self.transform:
            image, boxes = self.transform(image, boxes)

        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, width, height = box
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i
            
            if label_matrix[i, j, 20] == 0: # 해당 그리드에 객체가 없을 때만
                label_matrix[i, j, 20] = 1
                label_matrix[i, j, j:j+4] = torch.tensor([x_cell, y_cell, width, height])
                label_matrix[i, j, int(class_label)] = 1
        return image, label_matrix