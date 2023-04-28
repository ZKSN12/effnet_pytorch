import json
import cv2
import numpy as np 
import torch
import os
import torchvision.transforms as tvt
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from confusion_matrix import ConfusionMatrix
from dataloader import MyData

from PIL import Image
from model import efficientnet_b0 as create_model

def test(net, val_loader, class_indices):

    labels = [label for _,label in class_indices.items()]
    # creating an instance of confusion matrix
    confusion = ConfusionMatrix(num_classes=3, labels=labels)
    net.eval()
    with torch.no_grad():
        for i, val_data in enumerate(val_loader):
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            outputs = torch.softmax(outputs, dim=1)
            outputs = torch.argmax(outputs, dim=1)
            confusion.update(outputs.to("cpu").numpy(), val_labels.to("cpu").numpy())
    confusion.plot()    

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    data_transform = tvt.Compose(
        [tvt.Resize(224),
         tvt.CenterCrop(224),
         tvt.ToTensor(),
         tvt.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    #  read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indices = json.load(f)

    # class_of_interest = ['black_cherry', 'butternut', 'chestnut', 'red_oak', 'red_pine', 'walnut', 'white_oak', 'white_pine']
    class_of_interest = ['chestnut', 'red_oak','white_pine']
    batch_size = 1
    val_path = "C:/Users/kexin/Desktop/test3_0819"
    val_set = MyData(val_path, class_of_interest, data_transform)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False, num_workers=2)

    # create model
    model = create_model(num_classes=3).to(device)
    # load model weights
    model_weight_path = "C:/Users/kexin/Desktop/results_new/0914/results_0914_effnet/best_model.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    test(model, val_loader, class_indices)