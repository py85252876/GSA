import argparse
import numpy as np
import random
import torch

# https://github.com/lucidrains/imagen-pytorch.git
from imagen_pytorch import load_imagen_from_checkpoint, ImagenTrainer, ImagenConfig

from dataset_coco import *
from torchvision import transforms as T
from process_caption import *
from einops import rearrange
import os

# https://github.com/wandb/wandb.git
import wandb

import pickle


def parse():
    parser = argparse.ArgumentParser(description="Traing Imagen model.")
    parser.add_argument(
        "--model_dir", 
        type=str, 
        default="./exp1/",
        help="The directory that used to save checkpoints"
    )
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default=None,
        help="The directory that used to save data(image)"
    )
    parser.add_argument(
        "--project_name", 
        type=str, 
        default=None,
        help="The name that will show on wandb"
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
        default="./data/annotations/captions_train2017.json",
        help="annotation_file_train"
    )
    parser.add_argument(
        "--load_train_embedding", 
        type=str, 
        default="./embedding/text_base_train.pkl",
        help="embedding_file_train"
    )
    parser.add_argument(
        "--from_scratch",
        type=int,
        default=1,
        help="decided read from checkpoint or not"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="",
        help="read checkpoint from this dir"
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
    model_save_dir = args.model_dir
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)
    wandb.login()
    project_name = args.project_name
    hyperparams = {
        "steps": 600000,
        "dim": 128,
        "cond_dim": 512,
        "dim_mults": (1, 2, 4, 8),
        "image_sizes": [64, 256],
        "timesteps": 1000,
        "cond_drop_prob": 0.1,
        "batch_size": 32,
        'lr': 1e-4,
        'num_resnet_blocks': 3,
        "model_save_dir": model_save_dir,
        "dynamic_thresholding": True,
        "project_name": project_name,
        "train_embedding": args.load_train_embedding,
        "from_scratch": args.from_scratch,
        "checkpoint_path": args.checkpoint_path,
        "num_epochs": 400000,
        "annotation_file":args.annotation_file_train,
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


def make(config):
    train_embedding = check_embedding(config.train_embedding,config.annotation_file)
    data_transforms = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    imagen = ImagenConfig(
        unets = [dict(
        dim = config.dim, 
        cond_dim = 512,
        dim_mults = config.dim_mults, 
        num_resnet_blocks = config.num_resnet_blocks, 
        layer_attns = (False, False, False, True)
        ),dict(
        dim = config.dim, 
        cond_dim = 512,
        dim_mults = config.dim_mults, 
        num_resnet_blocks = (2,4,8,8), 
        layer_attns = (False, False, False, True),
        layer_cross_attns = (False, False, False, True)
        )],
        image_sizes = config.image_sizes,
        timesteps = config.timesteps,
        cond_drop_prob = config.cond_drop_prob,
        dynamic_thresholding = config.dynamic_thresholding
    ).create()

    imagen = imagen.cuda()
    trainer = ImagenTrainer(imagen, lr=config.lr, use_ema = True)

    ds = MSCOCODataset(
    root_dir=args.data_dir,
    embedding_file=train_embedding,
    transform=data_transforms
    )

    train_dataloader = torch.utils.data.DataLoader(ds, batch_size=config.batch_size, shuffle=True,num_workers = 8)
    print("finish trainer setting",flush = True)
    return trainer,train_embedding,train_dataloader

def train_unet_1(trainer, train_dataloader, config, sample_factor=None, validate_every=None, save_every=None,start_step = 0,embedding = None):
    assert config.model_save_dir[-1] == '/'
    num = 1
    i=0
    sample_every = 10
    for _ in range(config.num_epochs):
        for _, batch in enumerate(train_dataloader):
            loss = trainer(batch[0],text_embeds = batch[1],unet_number = 1,max_batch_size = 64)
            trainer.update(unet_number = 1)
            wandb.log({'train_loss': loss}, step=i+start_step)
            if sample_factor is not None and i % sample_every == 0 and trainer.is_main:
                sample = []
                image = trainer.sample(text_embeds=embedding.unsqueeze(0), cond_scale = 3. , return_pil_images = True,stop_at_unet_number = 1)
                sample.append(wandb.Image(image[0], caption="picture"))
                wandb.log({"samples": sample}, step=i+start_step)
                sample_every = int(sample_every * sample_factor )
            if save_every is not None and i != 0 and i % save_every == 0:
                trainer.save(f"{config.model_save_dir}unet{num}-{i+start_step}.pt")
            i+=1
    if save_every is not None and i % save_every != 0:
        trainer.save(f"{config.model_save_dir}unet{num}-{i+start_step}.pt")

def train_unet_2(trainer, train_dataloader,config, sample_factor=None, save_every=None, start_step = 0,embedding = None):
    assert config.model_save_dir[-1] == '/'
    print("now in train unet 2", flush = True)
    sample_every = 300
    num = 2
    for i in range(config.steps):
        loss = trainer.train_step(unet_number = 2,max_batch_size = 4)

        wandb.log({'train_loss': loss}, step=i+start_step)

        if sample_factor is not None and i % sample_every == 0 and trainer.is_main:
            image = trainer.sample(text_embeds=embedding.unsqueeze(0), cond_scale = 3. , return_pil_images = True,stop_at_unet_number = 2) 
            sample = []
            sample.append(wandb.Image(image[0], caption="picture"))
            wandb.log({"samples": sample}, step=i+start_step)
            sample_every = int(sample_every * sample_factor)
        if save_every is not None and i != 0 and i % save_every == 0:
            trainer.save(f"{config.model_save_dir}unet{num}-{i+start_step}.pt")
    if save_every is not None and i % save_every != 0:
        trainer.save(f"{config.model_save_dir}unet{num}-{i+start_step}.pt")

def main(hyperparams):
    
    with wandb.init(project=hyperparams['project_name'], config=hyperparams):
        
        config = wandb.config
        if hyperparams['from_scratch'] == 0:
            trainer,embedding,train_dataloader = make(config)
            print("start training",flush = True)
            train_unet_1(trainer, train_dataloader, config, sample_factor=1.3, save_every=50_000,embedding = embedding[0]['embedding'])
        elif hyperparams['from_scratch'] == 1:
            imagen = load_imagen_from_checkpoint(hyperparams['checkpoint_path'])
            imagen.timesteps = 20
            imagen.unets[1].layer_attns = (False, False, False, True)
            imagen.unets[1].layer_cross_attns = (False, False, False, True)
            print("start training",flush = True)
            print(imagen.timesteps,flush = True)
            print(imagen.unets[1].layer_attns,flush = True)
            trainer = ImagenTrainer(imagen, use_ema = True).cuda()
            train_embedding = check_embedding(args.load_train_embedding,args.annotation_file_train)
            data_transforms = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            ds = MSCOCODataset(
            root_dir=args.data_dir,
            embedding_file=train_embedding,
            transform=data_transforms
            )

            train_dataloader = torch.utils.data.DataLoader(ds, batch_size=config.batch_size, shuffle=True,num_workers = 8)
            trainer.add_train_dataloader(train_dataloader)
            print("finish trainer setting",flush = True)
            train_unet_1(trainer, config, sample_factor=1.3, save_every=50_000,start_step = 599999,embedding = train_embedding[0]['embedding'])
        elif hyperparams['from_scratch'] == 2:
            imagen = load_imagen_from_checkpoint(hyperparams['checkpoint_path'])
            imagen.timesteps = 20
            print("start training",flush = True)
            train_embedding = check_embedding(args.load_train_embedding,args.annotation_file_train)
            data_transforms = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            ds = MSCOCODataset(
            root_dir=args.data_dir,
            embedding_file=train_embedding,
            transform=data_transforms
            )

            train_dataloader = torch.utils.data.DataLoader(ds, batch_size=config.batch_size, shuffle=True,num_workers = 4)
            trainer = ImagenTrainer(imagen, use_ema = True).cuda()
            trainer.add_train_dataloader(train_dataloader)
            print("finish trainer setting",flush = True)
            train_unet_2(trainer, train_dataloader, config, sample_factor=1.7, save_every=5000,start_step = 0,embedding = train_embedding[0]['embedding'])

if __name__ == '__main__':
    args = parse()
    hyperparams=prepare(args)
    main(hyperparams)