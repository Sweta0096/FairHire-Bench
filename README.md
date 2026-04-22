# FairHire-Bench

[![ACM FAccT 2026](https://img.shields.io/badge/ACM%20FAccT-2026-blue)](https://facctconference.org/2026/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3805689.3806487-green)](https://doi.org/10.1145/3805689.3806487)

Official code for **"FairHireBench: A Cross-Generational Intersectional Bias Benchmark for Large Language Models in Automated Hiring"** — ACM FAccT 2026.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # add your API keys
```

## Dataset

The benchmark dataset (`dataset/file3.xlsx`) contains 10,005 candidate profiles across 15 intersectional groups (5 racial/ethnic × 3 gender categories). Place the file at `dataset/file3.xlsx` before running.

> The dataset is not included in this repository. Contact the authors to request access.

## Usage

```bash
python main.py
```

```bash
curl -X POST http://localhost:5316/api/v1/candidates \
  -H "Content-Type: application/json" \
  -d '{"models": ["gpt-4o-mini", "claude-3-5-haiku-20241022", "deepseek-chat"]}'
```

Results are saved as `{model_name}-results.csv`.

## Citation

```bibtex
@inproceedings{ratnani2026fairhirebench,
  author    = {Ratnani, Sweta Jaishankar and Li, Lingyao and Lou, Yitian and Li, Mingyang and Hua, Kaixun},
  title     = {FairHireBench: A Cross-Generational Intersectional Bias Benchmark for Large Language Models in Automated Hiring},
  booktitle = {Proceedings of the 2026 ACM Conference on Fairness, Accountability, and Transparency},
  series    = {FAccT '26},
  year      = {2026},
  location  = {Montreal, QC, Canada},
  publisher = {ACM},
  doi       = {10.1145/3805689.3806487}
}
```

## Contact

- Sweta Jaishankar Ratnani — University of South Florida
- Lingyao Li — University of South Florida
- Yitian Lou — University of Waterloo
- Mingyang Li — University of South Florida
- Kaixun Hua — University of South Florida
