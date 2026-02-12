import argparse
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import accuracy_score

from configs import get_config
from models.jumbo import Jumbo
from data_loading import ImageNetDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_size", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    kwargs = get_config(args.model_size)
    J =  kwargs.pop("J", None)
    model = Jumbo(args.gpu_id, J, **kwargs)
    weights = torch.load(args.model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(weights)
    model = model.eval()

    val_loader = DataLoader(
        ImageNetDataset(
            dataset=load_dataset("imagenet-1k", split="validation"),
            img_size=224,
        ),
        batch_size=512,
        shuffle=False,
        num_workers=8,
    )

    all_predictions = []
    all_labels = []
    for batch in val_loader:
        batch_images, batch_labels = batch
        batch_images = batch_images.to(args.gpu_id)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(batch_images)  # (batch_size, 1_000)

        batch_preds = logits.argmax(dim=-1).cpu().tolist()  # (batch_size)
        all_predictions += batch_preds
        all_labels += batch_labels.tolist()

    top_1 = accuracy_score(all_labels, all_predictions)
    print(f"Model path: {args.model_path}, Top-1 accuracy: {top_1}")
