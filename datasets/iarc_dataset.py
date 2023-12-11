import pandas as pd

import torch
import torch.nn as nn

import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, GaussianBlur, RandomEqualize, RandomInvert, ColorJitter

import PIL
from PIL import Image

from transformers import ViTImageProcessor

class IARCGeneralDataset(Dataset):
    '''
    Build dataset from address of image files.
    '''
    def __init__(
        self,
        annotations_file_path,
        mode='binary',
        labels2ids=None,
        aug_technique=None,
        keep_original_data=False
    ) -> None:

        data_ = None

        self.labels2ids = labels2ids
        
        self.data = None
        self.transform = transforms.Resize((400, 300))
        self.class_labels_strings = None

        self.data = pd.read_csv(annotations_file_path)
        self.class_labels = self.data['Diagnosis']

        if mode == 'binary':
            self.class_labels_strings = ['Normal', 'Abnormal']
        elif mode == 'multiclass':
            self.class_labels_strings = ['Normal', 'CIN1', 'CIN2', 'CIN3']
        else:
            print('Error establishing classification mode.')

        self.data['Diagnosis encoded'] = self.data['Diagnosis'].map(lambda x: self._encode_labels_into_ids(x))

    def _encode_labels_into_ids(self, row) -> int:
        if self.labels2ids is None:
            self.labels2ids = dict(zip(self.class_labels_strings, range(len(self.class_labels_strings))))
            
        for k, v in self.labels2ids.items():
            if k == row:
                return int(v)
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        case = self.data['Case Number'][idx]
        diagnosis = self.data['Diagnosis'][idx]
        image = self.transform(Image.open(f"{self.data['File'][idx]}"))

        return {'image': image,
                'label': diagnosis}

class collate_fn:
    def __init__(self):
        self.image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        
    def __call__(self, batch):
        images = [item['image']for item in batch]
        labels = [item['label'] for item in batch]

        # inputs = self.image_processor(images, return_tensors='pt')
        pixel_values = self.image_processor(images, return_tensors='pt')['pixel_values']
        
        return {'pixel_values': pixel_values,
                'labels': torch.LongTensor(labels)}
                # 'labels': labels}