import torch
import pickle
from torchvision import transforms as T
from torch.utils.data import Dataset
import json
from transformers import T5Tokenizer, T5EncoderModel
from einops import rearrange
import numpy as np
import random
'''
used to build embedding for text caption
'''


def encode_caption(caption,tokenizer, model):
        encoded = tokenizer.batch_encode_plus(
            caption,
            return_tensors="pt",
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True
        )
        with torch.no_grad():
            output = model(input_ids=encoded.input_ids , attention_mask=encoded.attention_mask)
            encoded_text = output.last_hidden_state.detach()

        attn_mask = encoded.attention_mask.bool()
        
        encoded_caption = encoded_text.masked_fill(~rearrange(attn_mask, '... -> ... 1'), 0.)
        return encoded_caption



def preprocess_captions(annotations_file, save_path):
    with open(annotations_file, 'r') as file:
        data = json.load(file)

    # model_name='google/t5-v1_1-base'
    # max_length=256
    # print("Loading tokenizer...",flush=True)
    # tokenizer = T5Tokenizer.from_pretrained(model_name, model_max_length=max_length)
    # print("Tokenizer loaded.", flush=True)

    # print("Loading model...", flush=True)
    # model = T5EncoderModel.from_pretrained(model_name)
    # print("Model loaded.", flush=True)
    # model.eval()

    print("start", flush=True)
    annotations = data['annotations']

    captions = []
    index = 0
    exist = []
    for annotation in annotations:
        if annotation['image_id'] in exist:
            continue
        else:
            exist.append(annotation['image_id'])
            temp = {}
            temp['image_id'] = annotation['image_id']
            temp['caption'] = annotation['caption']

            captions.append(temp)
            index += 1
            if index % 10000==0:
                print(index) 
    torch.save(captions,save_path)
    embeddings = []
    for i in range(0,len(captions), 100):
        print(f"start {i}", flush=True)
        batch_captions = captions[i:i + 100]
        encoded_caption_word = encode_caption([item['caption'] for item in batch_captions], tokenizer, model)
        for idx, item in enumerate(batch_captions):
            temp = {
                'image_id' : item['image_id'],
                'embedding': encoded_caption_word[idx]
            }
            embeddings.append(temp)
    print("finish saving annotation", flush=True)

    torch.save(embeddings ,save_path)


def main():
    torch.cuda.set_device(0)
    seed = 3128974198
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    preprocess_captions("./annotations/captions_train2017.json", "./caption/image_caption.pkl")
    # preprocess_captions("./data/annotations/captions_val2017.json", "./embedding/text_base_val.pkl")

if __name__ == '__main__':
    main()