import argparse
import inspect
import os

import torch
import torch.nn.functional as F

from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_dataset
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.utils import check_min_version
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
    ToTensor,
)
from tqdm.auto import tqdm


check_min_version("0.10.0.dev0")


logger = get_logger(__name__)


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def parse_args():
    parser = argparse.ArgumentParser(description="set dataset name")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset to train."
        ),
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "training dataset directory"
        ),
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Where we save the training and shadow model",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--which_l2", type=int, default=-1, help="-1 is the last checkpoint")
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "whether we will get the latest checkpoints"
        ),
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help=(
            "The directory that save the gradient information."
        ),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def main(args):
    accelerator = Accelerator()

    model = UNet2DModel(
        sample_size=args.resolution,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    prediction_type = "prediction_type" in set(inspect.signature(DDPMScheduler.__init__).parameters.keys())

    if prediction_type:
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.ddpm_num_steps,
            beta_schedule='linear',
            prediction_type=args.prediction_type,
        )
    else:
        noise_scheduler = DDPMScheduler(num_train_timesteps=args.ddpm_num_steps, beta_schedule='linear')

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0
    )

    augmentations = Compose(
        [
            Resize(args.resolution, interpolation=InterpolationMode.BILINEAR),
            CenterCrop(args.resolution),
            ToTensor(),
            Normalize([0.5], [0.5]),
        ]
    )
    if args.dataset_name is not None:
        dataset = load_dataset(
            args.dataset_name,
            None,
            cache_dir=None,
            split="train",
        )
    else:
        dataset = load_dataset("imagefolder", data_dir=args.train_data_dir, cache_dir=None, split="train")

    def transforms(examples):
        images = [augmentations(image.convert("RGB")) for image in examples["image"]]
        return {"input": images}

    dataset.set_transform(transforms)
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=0
    )

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )


    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)
    

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
            print(path)
        else:
            dirs = os.listdir(args.model_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[args.which_l2] 
            print(path)
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(args.model_dir, path))


    for epoch in range(0, 1):
        model.eval()
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        
        all_samples_grads = []
        for step, batch in enumerate(train_dataloader):
            clean_images = batch["input"]
            clean_images = clean_images.repeat(15,1,1,1)
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            timesteps = [100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1499] 
            timesteps = torch.tensor(timesteps, device = clean_images.device).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                model_output = model(noisy_images, timesteps).sample

                loss = F.mse_loss(model_output, noise)

                accelerator.backward(loss)
                gradients_l2_list = []
                for p in model.parameters():
                    gradients_l2_list.append(torch.norm(p.grad).unsqueeze(0))
                gradients_l2_list = torch.cat(gradients_l2_list)
                all_samples_grads.append(gradients_l2_list.unsqueeze(0))
                optimizer.zero_grad()
                progress_bar.update(1)

        progress_bar.close()

        accelerator.wait_for_everyone()        
        all_samples_grads = torch.cat(all_samples_grads)
        torch.save(all_samples_grads, args.output_name)
        break
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
