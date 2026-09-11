# X-GANet

PyTorch implementation of **X-GANet** (Basak et al., *Appl. Sci.* 2025) for graph-based network intrusion detection. The code follows Algorithm 1 (graph construction), Algorithms 2–3 (Cross-Diffused Attention), and Algorithm A1 (end-to-end training).

The training set is [NF-BoT-IoT-v2](packet_dataset/NF-BoT-IoT-v2.csv) (NetFlow v2 with source/destination IPs). A 90,600-flow stratified subsample is cached to `packet_dataset/nf_botiot_v2_sample.parquet` on first run.

## Requirements

```bash
pip install -r requirements.txt
```

On an NVIDIA A100, install a CUDA 12 PyTorch build, for example:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Train (A100)

From the project root:

```bash
python -m xganet.train --data packet_dataset/NF-BoT-IoT-v2.csv --sample-size 90600 --device cuda --amp bf16 --batch-size 128 --epochs 100 --num-workers 4
```

Best weights are written to `checkpoints/xganet_best.pt`. Training uses Adam (`lr=1e-4`, `weight_decay=5e-5`), categorical cross-entropy plus NT-Xent contrastive loss, and early stopping with patience 10.

## Evaluate

```bash
python -m xganet.evaluate --ckpt checkpoints/xganet_best.pt --data packet_dataset/nf_botiot_v2_sample.parquet
```

Reports accuracy, precision, recall, F1, ROC-AUC, a confusion matrix (`checkpoints/confusion_matrix.npy`), and mean fusion-gate / entropy-mask values.

## Hyperparameters (Table A1)

| Parameter | Symbol | Default |
| --- | --- | --- |
| Embedding dimension | d | 128 |
| XDA layers | L | 4 |
| Attention heads | H | 8 |
| Diffusion scale | — | 0.5 |
| Entropy threshold | τ | 1.2 |
| Contrastive temperature | τc | 0.1 |
| Contrastive weight | λ_contrast | 0.1 |
| Learning rate | α | 1e-4 |
| Weight decay | λ | 5e-5 |
| Batch size | B | 128 |
| LCM init scale | — | 0.1 |
| Early-stop patience | — | 10 epochs |
| Temporal window | Δt | 100 CSV row indices |

NF-BoT-IoT-v2 has no packet timestamps, so Δt uses the original CSV row index. IP edges follow Equation (4) (any shared endpoint).

## Tests

```bash
python -m pytest tests -q
```
