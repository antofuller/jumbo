import argparse
import os
from datasets import load_dataset
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, ToTensor, CenterCrop

from configs import get_config
from models.jumbo import Jumbo
from data_loading import ImageNetDataset, ResizeSmall


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_size", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="attn_maps")
    parser.add_argument("--num_images", type=int, default=50)
    args = parser.parse_args()

    kwargs = get_config(args.model_size)
    J = kwargs.pop("J", None)
    model = Jumbo(args.gpu_id, J, **kwargs)
    weights = torch.load(args.model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(weights)
    model = model.eval()

    img_size = 224
    dataset = load_dataset("imagenet-1k", split="validation")

    # normalized images for the model
    model_dataset = ImageNetDataset(dataset=dataset, img_size=img_size)

    # raw images for display (resize + crop only, no normalize)
    vis_transform = Compose([
        ToTensor(),
        ResizeSmall(img_size),
        CenterCrop((img_size, img_size)),
    ])

    os.makedirs(args.out_dir, exist_ok=True)

    for i in range(args.num_images):
        img_tensor, _ = model_dataset[i]
        raw_img = vis_transform(dataset[i]["image"].convert("RGB"))

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            attn = model.get_attn(img_tensor.unsqueeze(0).to(args.gpu_id))  # (1, h, w)

        attn = attn.float().squeeze(0)  # (h, w)
        attn = F.interpolate(attn.unsqueeze(0).unsqueeze(0), size=(img_size, img_size), mode="bilinear").squeeze()

        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        axes[0].imshow(raw_img.permute(1, 2, 0))
        axes[0].set_title("Image")
        axes[1].imshow(attn.cpu(), cmap="viridis")
        axes[1].set_title("Attention")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"{i:04d}.png"), dpi=150)
        plt.close()

    print(f"Saved {min(args.num_images, len(dataset))} attention maps to {args.out_dir}/")
