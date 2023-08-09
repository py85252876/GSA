from torch.utils.data import Dataset
import json
import os
from PIL import Image
from transformers import T5Tokenizer, T5EncoderModel
import torch
from einops import rearrange

class MSCOCODataset(Dataset):
    def __init__(self, root_dir,transform,embedding_file):
        self.root_dir = root_dir
        self.transform = transform
        self.preprocessed_data = embedding_file
        self.image_ids = [item['image_id'] for item in self.preprocessed_data]
        self.embedding_list = [item['embedding'] for item in self.preprocessed_data]

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_filename = os.path.join(self.root_dir, f"{image_id:012d}.jpg")
        image = Image.open(image_filename).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        encoded_captions = self.embedding_list[index].clone().detach() 

        return image, encoded_captions

    def __len__(self):
        return len(self.image_ids)