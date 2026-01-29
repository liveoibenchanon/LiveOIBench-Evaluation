LiveOIBench consists of 403 coding problems collected directly from the official websites of 72 competitions across 14 renowned Informatics Olympiads, focusing on contests held from 2023 onward. We collect all the official test cases, human contestant ranking results, and contestant Codeforces profiles.

This GitHub repo contains the evaluation toolkit for testing LLMs' solutions against the test cases and comparing their performance against human contestants.


## Installation

### Prerequisites

- Python **3.9+**
- `g++` (required for compiling C++ solutions and checkers)
- Linux environment (**strongly recommended**)
- *(Optional)* [`vllm`](https://github.com/vllm-project/vllm) for serving local models

### Setup
```bash
git clone 
cd LiveOIBench-Evaluation
pip install -r requirements.txt
```

> ⚠️ This repo is developed and tested on Linux.

---

## Data Setup

### Download from HuggingFace

The benchmark is hosted across three HuggingFace datasets:

- **Problems & Metadata**
- **Official Test Cases**
- **Human Contestant Data**

### Reconstruct the Dataset
```bash
export LIVEOIBENCH_ROOT=

python src/process_dataset.py \
  --download-dir "${LIVEOIBENCH_ROOT}/parquet_files" \
  --output-dir "${LIVEOIBENCH_ROOT}/data"
```

⏳ **Note:**
Reconstruction may take some time and disk space.
Total test cases exceed **30 GB**.

---

## Quick Start

### 1. Generate Solutions with an LLM

#### Start a local vLLM server
```bash
bash scripts/start_vllm.sh
```

#### Generate model solutions
```bash
bash scripts/run_model.sh
```

Outputs will be saved to:
```
${LIVEOIBENCH_ROOT}/predictions/<model>/
  ├── <model>_code.json
  └── <model>_raw.json
```

---

### 2. Judge Model Solutions

Run judging against official test cases:
```bash
bash scripts/run_model_solutions.sh
```

Results will be saved to:
```
${LIVEOIBENCH_ROOT}/evaluation/submission_results/<model>/<model>_<timestamp>.json
```

---

### 3. Compare Against Human Contestants

Compute rankings across:
- Individual contests
- Olympiad competitions
- Overall benchmark performance
```bash
bash scripts/generate_all_ranking.sh
```

This generates CSV summaries with:
- Contest-level scores
- Relative human percentile
- Aggregate benchmark results

---

## Acknowledgements

We thank Codeforces and the organizing committees of Informatics Olympiads worldwide for making problem materials and contest data publicly available.