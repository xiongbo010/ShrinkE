
Shrinking Embedding

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

![log](https://github.com/xiongbo010/ShrinkE/blob/main/log.png)
