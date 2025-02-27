# Simpler Fast Vision Transformers with a Jumbo CLS Token

**Paper:** [https://arxiv.org/abs/2502.15021](https://arxiv.org/abs/2502.15021)

**Download weights:**

```bash
wget https://huggingface.co/antofuller/jumbo/resolve/main/jumbo_pico_i1k_finetuned_224.pt
wget https://huggingface.co/antofuller/jumbo/resolve/main/jumbo_nano_i1k_finetuned_224.pt
wget https://huggingface.co/antofuller/jumbo/resolve/main/jumbo_tiny_i1k_finetuned_224.pt
wget https://huggingface.co/antofuller/jumbo/resolve/main/jumbo_small_i1k_finetuned_224.pt
```

**Run ImageNet-1K evals:**
```bash
pip install -r requirements.txt
python eval_i1k_script.py
```
