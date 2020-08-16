import os
import pandas as pd
from torch.utils.data import Dataset
import PIL

# Диагноз: пневма - 1, норма - 0
class Xray_dataset(Dataset):
    def __init__(self, what='train', transform=None):
        assert what == 'train' or what == 'test' or what == 'val'

        self.transform = transform

        self.data = pd.DataFrame(columns=['Path', "Diagnosis"])

        # root = '../chest_xray'
        root = '/content/drive/My Drive/Colab Notebooks/Chest-Xray/data'

        root = os.path.join(root, what)

        pneum = os.path.join(root, "PNEUMONIA")

        for filename in os.listdir(pneum):
            self.data = self.data.append({
                'Path': os.path.join(pneum, filename),
                'Diagnosis': 1
            }, ignore_index=True)

        norm = os.path.join(root, "NORMAL")

        for filename in os.listdir(norm):
            self.data = self.data.append({
                'Path': os.path.join(norm, filename),
                'Diagnosis': 0
            }, ignore_index=True)

        self.data = self.data.sample(frac=1)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        root_image, label = self.data.iloc[index]

        image = PIL.Image.open(root_image).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label