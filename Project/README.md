# LlamaFactory — Replication Study
### Unified Efficient Fine-Tuning of 100+ Language Models

> Partial replication of **Zheng et al. (2024)** — *LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models* (ACL 2024 System Demonstrations) 

---

## Overview

This project replicates the qualitative patterns of **Table 4** from the LlamaFactory paper on a smaller model and an instruction-following task. We compare five efficient fine-tuning methods on **TinyLlama-1.1B-Chat-v1.0** using the **Alpaca GPT-4** dataset, and contribute a novel data-size ablation not present in the original paper.

| File | Description |
|---|---|
| `llamaFactory_experiments.ipynb` | Full experimental notebook (training, evaluation, plots, tables) |
| `LlamaFactory_Report.pdf` | Written report (framework analysis, methods, results, contributions) |
| `LlamaFactory_Presentation.pdf` | Beamer presentation slides |

---

## Experimental Setup

| Parameter | Value |
|---|---|
| **Model** | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
| **Dataset** | Alpaca GPT-4 (`llamafactory/alpaca_gpt4_en`) — 10 000 examples |
| **Validation split** | 10% held-out, seeded (seed = 42) |
| **Sequence length** | 512 tokens |
| **Precision** | fp16 |
| **Chat template** | Zephyr (auto-selected by LlamaFactory for TinyLlama-Chat) |
| **Checkpoint selection** | Best validation loss (`load_best_model_at_end = true`) |
| **GPU** | **NVIDIA RTX 6000 Ada Generation (50.9 GB VRAM)** |

---

## Methods Compared

| Method | Trainable Params | Strategy |
|---|---|---|
| **Baseline** | — | Pre-trained model, zero-shot perplexity only |
| **Full fine-tuning** | 1 100M (all) | All parameters updated |
| **Freeze-tuning** | 176M (last 4 blocks) | Earlier layers frozen |
| **GaLore** | 1 100M (all) | Full params, gradients projected to low-rank subspace |
| **LoRA** | 18M | Low-rank adapter matrices (rank 64, α 128) |
| **QLoRA** | 18M | LoRA on 4-bit NF4 quantized base model |

---

## Results

### Table 1 — Main Experiment (10 000 examples)

| Method | Memory (GB) | Throughput (tok/s) | Best Eval Loss | PPL |
|---|---|---|---|---|
| Baseline | — | — | — | 3.51 |
| Full-FT | 24.35 | 6 981 | 1.109 | 3.03 |
| Freeze | 9.81 | 16 836 | 1.134 | 3.11 |
| GaLore | 22.04 | 6 838 | 1.123 | 3.07 |
| **LoRA** | **8.02** | 10 440 | **1.117** | **3.06** |
| **QLoRA** | **8.02** | 10 454 | **1.117** | **3.06** |

### Table 2 — Novel Ablation: LoRA vs. QLoRA Across Data Sizes

| Examples | LoRA PPL | QLoRA PPL | Gap (Q − L) |
|---|---|---|---|
| 100 | 3.18 | 3.18 | −0.0003 |
| 500 | 3.28 | 3.28 | ≈ 0 |
| 1 000 | 3.05 | 3.05 | ≈ 0 |
| 2 000 | 2.93 | 2.93 | ≈ 0 |
| 5 000 | 3.03 | 3.03 | ≈ 0 |
| 10 000 | 3.06 | 3.06 | ≈ 0 |

**Finding:** at 1.1B parameter scale, LoRA and QLoRA are statistically indistinguishable across all dataset sizes tested. The quality gap documented in the original paper (on 2–13B models) does not appear at this scale.

---

## GPU Requirements

> **This project was run on an NVIDIA RTX 6000 Ada Generation (50.9 GB VRAM).**

If you run this notebook on a lower-memory GPU such as the **Google Colab Tesla T4 (16 GB)**, be aware of the following:

- **Full fine-tuning (Full-FT) will fail with OOM** — updating all 1.1B parameters in fp16 requires approximately 24 GB of peak GPU memory (weights + Adam optimizer states), which exceeds the T4's capacity.
- **GaLore will also fail with OOM** — although GaLore compresses optimizer states via low-rank gradient projection, it still updates all parameters and requires approximately 22 GB of peak memory, also beyond what the T4 can provide.
- **LoRA, QLoRA, and Freeze-tuning will run without issues** on a T4 — their peak memory usage is between 8 and 10 GB, well within the T4's 16 GB limit.

To run the full comparison on a T4, you can either reduce `MAX_MAIN_SAMPLES` (e.g. to 2 000) and lower the batch size, or simply skip the Full-FT and GaLore configs and focus on the three memory-efficient methods.

---

## Reference

```bibtex
@inproceedings{zheng2024llamafactory,
  title     = {{LlamaFactory}: Unified Efficient Fine-Tuning of 100+ Language Models},
  author    = {Zheng, Yaowei and Zhang, Richong and Zhang, Junhao and Ye, Yanhan and
               Luo, Zheyan and Feng, Zhangchi and Ma, Yongqiang},
  booktitle = {Proceedings of the 62nd Annual Meeting of the Association for
               Computational Linguistics (Volume 3: System Demonstrations)},
  pages     = {400--410},
  year      = {2024},
}
```
