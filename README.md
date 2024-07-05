
# Shrinking Embedding for Hyper-Relational Knowledge Graphs
[![Identifier](https://img.shields.io/badge/doi-10.18419%2Fdarus--3978-d45815.svg)](https://doi.org/10.18419/darus-3978)

## Requirements
Create a new conda environment and execute `setup.sh`.
Alternatively
```
pip install -r requirements.txt
```

### Starting training and evaluation
It is advised to run experiments on a GPU otherwise training might take long.
Use `DEVICE cuda` to turn on GPU support, default is `cpu`.
Don't forget to specify `CUDA_VISIBLE_DEVICES` before `python` if you use `cuda`

nohup python -u run.py DEVICE cuda DATASET jf17k > logs/nohup_shrink_jf17k_200_new.out  2>&1 &

You should obtain the following results after training (see logging file at logs/nohup_shrink_jf17k_200_new.out)
## Contact

Contact: xiongbo010@gmail.com
