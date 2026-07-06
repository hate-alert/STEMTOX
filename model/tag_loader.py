from torch.utils.data import Dataset
from PIL import Image
import os
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

class TagDataset(Dataset):
    def __init__(self, df, dataset_tag):
        self.annotations = df
        self.dataset_tag = dataset_tag

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        try:
            image = Image.open(f'{self.annotations.loc[index, "img"]}').convert("RGB")
        
        except Exception as e:
            image = ''
            print('Image not found!!')
        if self.dataset_tag == "fhm" or self.dataset_tag == "mami":
            text = (
            "Generate tags by considering image and ocr \n"
            f"Ocr text associated with the image: {self.annotations.loc[index, 'text']}\n"
        )
            label = self.annotations.loc[index, "generated_tags"]

        else:
            text = (
                "Generate tags by considering image, ocr and title\n"
                f"Ocr text associated with the image: {self.annotations.loc[index, 'ocr']}\n"
                f"Title associated with the image: {self.annotations.loc[index,'title']}\n"
            )
            label = self.annotations.loc[index, "tags"]
        class_label = torch.tensor(self.annotations.loc[index, "finegrained"])
        return image, text, label, class_label
