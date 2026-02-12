def get_config(size):
    assert size in ["pico", "nano", "tiny", "small", "base"]

    if size == "pico":
        return {
            "img_size": 224,
            "num_classes": 1_000,
            "embed_dim": 96,
            "depth": 12,
            "num_heads": 3,
            "J": 6
        }
    elif size == "nano":
        return {
            "img_size": 224,
            "num_classes": 1_000,
            "embed_dim": 128,
            "depth": 12,
            "num_heads": 4,
            "J": 6
        }
    elif size == "tiny":
        return {
            "img_size": 224,
            "num_classes": 1_000,
            "embed_dim": 192,
            "depth": 12,
            "num_heads": 3,
            "J": 6
        }
    elif size == "small":
        return {
            "img_size": 224,
            "num_classes": 1_000,
            "embed_dim": 384,
            "depth": 12,
            "num_heads": 6,
            "J": 6
        }
    else:
        # base
        return {
            "img_size": 224,
            "num_classes": 1_000,
            "embed_dim": 768,
            "depth": 12,
            "num_heads": 12,
            "J": 3
        }