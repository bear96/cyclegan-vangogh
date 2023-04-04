from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import glob
from PIL import Image
import random
import os
from utils import convert_to_rgb
from hyperparameters import hp 

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned        

        self.files_A = sorted(glob.glob(os.path.join(root, "%sA" % mode) + "/*.*")) #van gogh
#         self.files_B = sorted(glob.glob(os.path.join(root, "%sB" % mode) + "/*.*"))
        self.files_B = sorted(glob.glob( "/kaggle/input/human-faces/Humans"+ "/*.*")) #pics

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])

        if self.unaligned:
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)])

        # Convert grayscale images to rgb
        if image_A.mode != "RGB":
            image_A = convert_to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = convert_to_rgb(image_B)

        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        
        # Finally ruturn a dict
        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
    

root_path = ""

train_transforms_ = [
    transforms.Resize((286, 286)),
    transforms.RandomRotation(degrees=(0,180)),
    transforms.RandomCrop(size=(hp.img_size,hp.img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

val_transforms_ = [
    transforms.Resize((hp.img_size, hp.img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

train_dataloader = DataLoader(
    ImageDataset(root_path, mode=hp.dataset_train_mode, transforms_=train_transforms_),
    batch_size=hp.batch_size,
    shuffle=True,
    num_workers=2,
)
val_dataloader = DataLoader(
    ImageDataset(root_path, mode=hp.dataset_test_mode, transforms_=val_transforms_),
    batch_size=8,
    shuffle=True,
    num_workers=2,
)