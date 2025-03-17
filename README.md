# Anonymous Paper Submission: Large Language Model Unlearning for Source Code

This repository contains the code and data for the anonymous submission titled **Large Language Model Unlearning for Source Code**.


## Instructions for Reproducibility

To reproduce the results of this study, follow these steps:

### 1. Set Up the Environment
Install the required dependencies using `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 2. Run the Experiments
Execute the `run.sh` script to run the experiments:
```bash
bash run.sh
```

## Experimental Results

Below are the key experimental results from our work:

### 1. Main Results
<div align="center">
  <img src="https://github.com/jiangxxxue/PROD/blob/main/figures/TradeOff.png" alt="Main Results" width="100%" />
</div>
<p align="center">
  <b>Figure 1: Forget quality versus model utility across different downstream tasks. Points for all unlearning approaches are plotted at identical epochs during the later stages of training.</b>
</p>

### 2. Application on Different LLMs
<div align="center">
  <img src="https://github.com/jiangxxxue/PROD/blob/main/figures/DifferentLLMs.png" alt="Application on Different LLMs" width="50%" />
</div>
<p align="center">
  <b>Figure 2: The performance of PROD on different LLMs.</b>
</p>

### 3. Adversarial Attacks
<div align="center">
  <img src="https://github.com/jiangxxxue/PROD/blob/main/figures/Attack.png" alt="Adversarial Attacks" width="50%" />
</div>
<p align="center">
  <b>Figure 3: Comparison of adversarial attack results across different LLM unlearning approaches, showing mean attack effects with maximum and minimum ranges.</b>
</p>

### 4. Ablation Study
<div align="center">
  <img src="https://github.com/jiangxxxue/PROD/blob/main/figures/AblationStudy.png" alt="Ablation Study" width="100%" />
</div>
<p align="center">
  <b>Figure 4: Ablation results on alternative loss function, and the impact of hyperparameters top_p and α in PROD.</b>
</p>
