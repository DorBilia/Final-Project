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

## Stream (replay alerts)

Training and evaluate are unchanged. `python -m xganet.stream` reads a NetFlow CSV or parquet **in row order**, scores a sliding window with a trained checkpoint, and prints an alert for each flow whose predicted class is not Benign and whose softmax probability is at least `--threshold`.

This is **offline-trained, online-replay**. There is no packet sniffer or nProbe/Zeek feeder in this pass. Prefer the cached sample parquet for demos (correct columns plus `row_idx`). The full 37.8M-row CSV is supported via chunked scan; use `--max-windows` to cap a run.

```bash
python -m xganet.stream \
  --ckpt checkpoints/xganet_best.pt \
  --data packet_dataset/nf_botiot_v2_sample.parquet \
  --window 128 --stride 128 --threshold 0.5 --rate 0
```

| Flag | Default | Role |
| --- | --- | --- |
| `--window` | 128 | Flows in each scored graph (same size as training `B`) |
| `--stride` | 128 | How far to advance; `32` overlaps windows for snappier alerts |
| `--threshold` | 0.5 | Minimum predicted-class probability to alert |
| `--rate` | 0 | Ingest pace in flows/sec; `0` is as fast as the GPU allows |
| `--jsonl` | unset | Append the same alert records as JSON lines |
| `--max-windows` | unset | Stop after this many scored windows |
| `--benign-class` | `Benign` | Required if the checkpoint has no class named `Benign` |

A leftover partial window is scored if it has at least two flows; a single leftover row is skipped.

**Scaling is frozen** from the checkpoint (`feat_min` / `feat_max`). The stream never refits min–max.

**Structure features are window-local.** Degree, centrality, and IP/time flags are computed on the current window only, not on the 90,600-row training subsample. That is the honest online version of Table 4 and will differ slightly from training’s global degrees. Time edges still use CSV `row_idx` and `Δt` (default 100 rows).

Alert line example:

```
ALERT row=1842 192.168.1.10 -> 10.0.0.5 pred=DDoS p=0.9731 gt=DDoS
```

## Tests

```bash
python -m pytest tests -q
```
