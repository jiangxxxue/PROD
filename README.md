# Anonymous Paper Submission: Large Language Model Unlearning for Source Code

This repository contains the code, data, and supplementary materials for the anonymous submission titled **Large Language Model Unlearning for Source Code**.


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

Below are the key experimental results from our study:

### 1. Main Results
![Main Results](https://github.com/jiangxxxue/PROD/blob/main/figures/TradeOff.png)  
*Figure 1: Forget quality versus model utility across different downstream tasks. Points for all unlearning approaches are plotted at identical epochs during the later stages of training.*

### 2. Application on Different LLMs
![Ablation Study]([https://github.com/username/repository/blob/main/figures/ablation_study.png](https://github.com/jiangxxxue/PROD/blob/main/figures/DifferentLLMs.png))  
*Figure 2: The performance of PROD on different LLMs.*

### 3. Adversarial Attacks
![Scalability Analysis](https://github.com/jiangxxxue/PROD/blob/main/figures/Attack.png)  
*Figure 3: Comparison of adversarial attack results across different LLM unlearning approaches, showing mean attack effects with maximum and minimum ranges.*

### 4. Ablation Study
![Case Study](https://github.com/jiangxxxue/PROD/blob/main/figures/AblationStudy.png)  
*Figure 4: Ablation results on alternative loss function, and the impact of hyperparameters $\mathbf{top\_p}$ and $\mathbf{\alpha}$ in PROD.*
