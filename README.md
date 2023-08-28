# MiraMark
Benchmarking suite for machine learning workloads.

## Installation
```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Running
The amount of runs to do is specified by the `MIRAMARK_RUNS` environment variable. The default is 10. Running `main.py` will run all benchmarks and output the results to `results.md`. **Please note that these tests are very computationally intensive and may take a long time to run.**

```bash
source env/bin/activate
MIRAMARK_RUNS=3 python3 main.py
```