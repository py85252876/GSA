import argparse
import inspect
import math
import os

import torch
import torch.nn.functional as F

# https://github.com/huggingface/accelerate.git
from accelerate import Accelerator
from accelerate.logging import get_logger

# https://github.com/huggingface/datasets.git
from datasets import load_dataset

# https://github.com/huggingface/diffusers.git
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
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


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "training data directory."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=True,
        help="output model directory",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
        help=(
            "The resolution of input image."
        ),
    )
    parser.add_argument(
        "--train_batch_size", 
        type=int, 
        default=16, 
        help="Batch size for the training dataloader."
    )
    parser.add_argument(
        "--eval_batch_size", 
        type=int, 
        default=16, 
        help="The number of images to generate for evaluation."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "The number of subprocesses to use for data loading. "
        ),
    )
    parser.add_argument(
        "--num_epochs", 
        type=int, 
        default=200, 
        help="Total training epoch"
        )
    parser.add_argument(
        "--save_images_epochs", 
        type=int, 
        default=10, 
        help="How often to save images during training."
        )
    parser.add_argument(
        "--save_model_epochs", 
        type=int, 
        default=100, 
        help="How often to save the model during training."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            "The scheduler type to use. "
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", 
        type=int, 
        default=500, 
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        default=True,
        help="Whether to use Exponential Moving Average for the final model weights.",
    )
    parser.add_argument(
        "--ema_inv_gamma", 
        type=float, 
        default=1.0, 
        help="The inverse gamma value for the EMA decay."
    )
    parser.add_argument(
        "--local_rank", 
        type=int, 
        default=-1, 
        help="For distributed training: local_rank"
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"]
    )
    parser.add_argument(
        "--diffusion_steps", 
        type=int, 
        default=1000
    )
    parser.add_argument(
        "--diffusion_beta_schedule", 
        type=str, 
        default="linear"
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    return args


def main(args):
    logging_dir = os.path.join(args.output_dir, "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        logging_dir=logging_dir,
    )

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
    accepts_prediction_type = "prediction_type" in set(inspect.signature(DDPMScheduler.__init__).parameters.keys())

    if accepts_prediction_type:
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.diffusion_steps,
            beta_schedule=args.diffusion_beta_schedule,
            prediction_type=args.prediction_type,
        )
    else:
        noise_scheduler = DDPMScheduler(num_train_timesteps=args.diffusion_steps, beta_schedule=args.diffusion_beta_schedule)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.95, 0.999),
        weight_decay=1e-6,
        eps=1e-08,
    )

    augmentations = Compose(
        [
            Resize(args.resolution, interpolation=InterpolationMode.BILINEAR),
            CenterCrop(args.resolution),
            ToTensor(),
            Normalize([0.5], [0.5]),
        ]
    )
    dataset = load_dataset("imagefolder", data_dir=args.train_data_dir, split="train")
    def transforms(examples):
        images = [augmentations(image.convert("RGB")) for image in examples["image"]]
        return {"input": images}

    logger.info(f"Dataset size: {len(dataset)}")

    dataset.set_transform(transforms)
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    accelerator.register_for_checkpointing(lr_scheduler)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    ema_model = EMAModel(
        accelerator.unwrap_model(model),
        inv_gamma=args.ema_inv_gamma,
        power=0.75,
        max_value=0.9999,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    global_step = 0
    epoch_saved = -1
    for epoch in range(args.num_epochs):
        model.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            clean_images = batch["input"]
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bsz = clean_images.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device
            ).long()

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                model_output = model(noisy_images, timesteps).sample

                loss = F.mse_loss(model_output, noise)

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                if args.use_ema:
                    ema_model.step(model)
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                

                if epoch % args.save_model_epochs == 0 and epoch != 0 and epoch != epoch_saved:
                    print(epoch)
                    print(epoch_saved)
                    epoch_saved = epoch
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            if args.use_ema:
                logs["ema_decay"] = ema_model.decay
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
        progress_bar.close()

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            if epoch % args.save_images_epochs == 0 or epoch == args.num_epochs - 1:
                pipeline = DDPMPipeline(
                    unet=accelerator.unwrap_model(ema_model.averaged_model if args.use_ema else model),
                    scheduler=noise_scheduler,
                )

                generator = torch.Generator(device=pipeline.device).manual_seed(0)
                images = pipeline(
                    generator=generator,
                    batch_size=args.eval_batch_size,
                    output_type="numpy",
                ).images

                images_processed = (images * 255).round().astype("uint8")

                accelerator.get_tracker("tensorboard").add_images(
                    "test_samples", images_processed.transpose(0, 3, 1, 2), epoch
                )

        accelerator.wait_for_everyone()
        
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        accelerator.save_state(save_path)
        logger.info(f"Saved state to {save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)