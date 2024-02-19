import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

data_loc = "./data/train"
test_data_loc = "./data/test"

dataset = ImageFolder(data_loc, transform=transforms.Compose([
    transforms.Resize((200,200)),transforms.ToTensor()
    ]))

test_dataset = ImageFolder(test_data_loc, transform=transforms.Compose([
    transforms.Resize((200,200)),transforms.ToTensor()
    ]))

def display_img(img, label):
    print(f"Label: {dataset.classes[label]}")
    plt.imshow(img.permute(1,2,0))
    plt.show()

input = int(input("enter num: "))

display_img(*dataset[input])