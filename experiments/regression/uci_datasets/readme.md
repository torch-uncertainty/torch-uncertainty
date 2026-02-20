# UCI Regression - Benchmark


| Dataset        | Number of Instances | Number of Features |
| -------------- | ------------------- | ------------------ |
| Boston Housing | 506                 | 13                 |


> [!WARNING]
> Some datasets require installing additional packages.


This folder contains the code to train models on the UCI regression datasets. The task is to predict (a) continuous target variable(s).

**General command to train a model:**

```bash
cd experiments/regression/uci_datasets
python mlp.py fit --config configs/{dataset}/{network}/{dist_family}.yaml
```

*Example:*

```bash
cd experiments/regression/uci_datasets
python mlp.py fit --config configs/boston/mlp/laplace.yaml
```
