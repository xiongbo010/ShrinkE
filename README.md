
SHrinking Embedding

## Requirements
* Python 3.7
* PyTorch 1.5.1
* torch-geometric 1.6.1
* torch-scatter 2.0.5
* tqdm
* wandb

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
