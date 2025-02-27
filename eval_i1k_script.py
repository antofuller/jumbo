from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import accuracy_score

from configs import get_config
from models.jumbo import Jumbo
from data_loading import ImageNetDataset


sizes = ["pico", "nano", "tiny", "small"]
gpu_id = 0

for size in sizes:
    kwargs = get_config(size)
    model = Jumbo(gpu_id, J=6, jumbo_mlp_ratio=4, **kwargs)
    weights = torch.load(f"jumbo_{size}_i1k_finetuned_224.pt")
    model.load_state_dict(weights)
    model = torch.compile(model)  # optional for speedup
    model = model.eval()

    val_dataset = ImageNetDataset(
        dataset=load_dataset("imagenet-1k", split="validation"),
        do_augment=False,
        img_size=224,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=8,
    )

    all_predictions = []
    all_labels = []
    for batch in val_loader:
        batch_images, batch_labels = batch
        batch_images = batch_images.to(gpu_id)

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(batch_images)  # (batch_size, 1_000)

        batch_preds = logits.argmax(dim=-1).cpu().tolist()  # (batch_size)
        all_predictions += batch_preds
        all_labels += batch_labels.tolist()

    top_1 = accuracy_score(all_labels, all_predictions)
    print(f"Size: {size}, Top-1 accuracy: {top_1}")
