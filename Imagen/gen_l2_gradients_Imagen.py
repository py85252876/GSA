import argparse
import numpy as np
import random
import torch

# https://github.com/lucidrains/imagen-pytorch.git
from imagen_pytorch.utils import load_imagen_from_checkpoint
from imagen_pytorch.configs import ImagenConfig
from imagen_pytorch.trainer import ImagenTrainer

from dataset_MSCOCO import *
from torchvision import transforms as T
from prepare_caption import *
from einops import rearrange
import os

# https://github.com/wandb/wandb.git
import wandb

import pickle
from tqdm import tqdm


def parse():
    parser = argparse.ArgumentParser(description="Generate l2-norm gradient information form model.")
    parser.add_argument(
        "--gradient_path", 
        type=str, 
        default="./exp1/",
        help="The directory that used to save gradients."
    )
    parser.add_argument(
        "--data_dir",
        type=str, 
        default=None,
        help="The directory that used to save data(image)."
    )
    parser.add_argument(
        "--gpu_id", 
        type=int, 
        default=0,
        help="The gpu number to run the code."
    )
    parser.add_argument(
        "--annotation_file_train", 
        type=str, 
        default="./data/annotations/captions_train2017.json"
    )
    parser.add_argument(
        "--load_train_embedding", 
        type=str, 
        default="./embedding/text_base_train.pkl"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="",
        help="read checkpoint from this dir."
    )
    parser.add_argument(
        "--get_unet",
        type=int,
        default=1,
        help="unet number to get gradient."
    )
    
    parser.add_argument(
        "--attack_method",
        type=int,
        default=1,
        help="GSA method number."
    )
    args = parser.parse_args()
    return args

def prepare(args):
    seed = 3128974198
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    hyperparams = {
        "steps": 600000,
        "dim": 128,
        "cond_dim": 512,
        "dim_mults": (1, 2, 4, 8),
        "image_sizes": [64, 256],
        "timesteps": 1000,
        "cond_drop_prob": 0.1,
        "batch_size": 36,
        'lr': 1e-4,
        'num_resnet_blocks': 3,
        "dynamic_thresholding": True,
        "train_embedding": args.load_train_embedding,
        "checkpoint_path": args.checkpoint_path,
        "num_epochs": 400,
        "gradient_path": args.gradient_path,
        "get_unet":args.get_unet,
        "annotation_file":args.annotation_file_train,
        "attack_method" : args.attack_method,
    }
    torch.cuda.set_device(args.gpu_id)
    return hyperparams

def check_embedding(embedding_file,annotation_file):
    if os.path.isfile(embedding_file):
        print("load file",flush = True)
        data = torch.load(embedding_file)
        print('File exists and data is loaded.',flush = True)
        return data
    else:
        print('File does not exist.',flush = True)
        return preprocess_captions(annotation_file,embedding_file)


def gen_gradients(trainer, valid_dataloader,config):
    i=0
    progress_bar = tqdm(total=len(valid_dataloader))
    progress_bar.set_description(f"Data number {i}")
    for epoch in range(0,1):
        trainer.imagen.unets[config['get_unet']-1].eval()
        all_gradient_list = []
        print(f"get unet gradient from unet {config['get_unet']}",flush = True)
        for _, batch in enumerate(valid_dataloader):
            gradients_l2_list = trainer.get_gradient(batch[0],text_embeds = batch[1],unet_number = config['get_unet'],max_batch_size = 1,attack_method = config['attack_method'])
            all_gradient_list.append(gradients_l2_list)
            progress_bar.update(1)
        all_gradient_list = torch.stack(all_gradient_list,dim=0)
        print(all_gradient_list.shape)
        progress_bar.close()
        torch.save(all_gradient_list,config['gradient_path'])

def main(hyperparams):
    print("start running..", flush=True)
    config = hyperparams
    imagen = load_imagen_from_checkpoint(hyperparams['checkpoint_path'])
    train_embedding = check_embedding(hyperparams['train_embedding'],hyperparams['annotation_file'])
    data_transforms = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tst_ds = MSCOCODataset(
    root_dir=args.data_dir,
    embedding_file=train_embedding[:5000],
    transform=data_transforms
    )
    valid_dataloader = torch.utils.data.DataLoader(tst_ds, batch_size=1, shuffle=False, num_workers = 4)
    trainer = ImagenTrainer(imagen).cuda()
    trainer.add_valid_dataloader(valid_dataloader)
    print("finish trainer setting",flush = True)
    gen_gradients(trainer, valid_dataloader, config)
        
if __name__ == '__main__':
    print("1",flush=True)
    args = parse()
    print("1",flush=True)
    hyperparams=prepare(args)
    print("1",flush=True)
    main(hyperparams)