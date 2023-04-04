from torch.utils.data import Dataset
import torchvision.transforms as transforms
import glob
from PIL import Image
import random
from utils import convert_to_rgb

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.root_A = root[0]
        self.root_B = root[1]       

        self.files_A = sorted(glob.glob(self.root_A + "/*.*")) #van gogh
        # self.files_B = sorted(glob.glob( "/kaggle/input/human-faces/Humans"+ "/*.*")) #pics
        self.files_B = sorted(glob.glob(self.root_B + "/*.*")) #pics

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
    