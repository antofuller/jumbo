import argparse
import time
import torch

from configs import get_config
from models.jumbo import Jumbo


def test_time(model, batch):
    N = 50
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(N):
            _ = model(batch)

        start_time = time.time()
        for _ in range(N):
            _ = model(batch)
        diff = time.time() - start_time

    imgs_per_sec = (N * batch.shape[0]) / diff
    return imgs_per_sec


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    kwargs = get_config(args.model_size)
    J = kwargs.pop("J", None)
    model = Jumbo(args.gpu_id, J, **kwargs).eval()
    model = torch.compile(model)

    batch = torch.randn(256, 3, kwargs["img_size"], kwargs["img_size"]).to(args.gpu_id)
    imgs_per_sec = test_time(model, batch)
    print(f"Model: {args.model_size}, Speed: {imgs_per_sec:.1f} imgs/sec")
