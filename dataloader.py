import os
from matplotlib import pyplot as plt
from PIL import Image
import torch
import glob
from torch.utils.data import Dataset
import torchvision.transforms as tvt
from torch.utils.data import DataLoader, Dataset


class MyData(Dataset):
    def __init__(self, root, class_list=None, transform = None):
        self.root = root
        # load classes
        if class_list == None:
            new_class_list = []
            for cla in os.listdir(self.root):
                new_class_list.append(cla)
            self.class_list = new_class_list
        else:
            self.class_list = class_list
        # load transforms
        if transform == None:
            transform = tvt.Compose([
                tvt.ToTensor(),
                tvt.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
        else:
            self.transform = transform
        
        self.path_list = []  # a list of image paths
        self.img_list = []   # a list of images paths with corresponding labels

        for file in self.class_list:
            file_path = os.path.join(self.root, file, "")
            self.path_list.append(file_path)
            file_label = self.class_list.index(file)
            pattern = file_path + '*'
            for img in glob.glob(pattern):
                img_list = [file_label, img]
                # add image label and path to a image list
                self.img_list.append(img_list)
    
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_label = self.img_list[index][0]
        img = Image.open(self.img_list[index][1])
        img_transformed = self.transform(img)
        return img_transformed, img_label
    
    


