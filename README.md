# gpuserver-hacker
how to hold gpu cards on multi-gpu servers, code contributed by Haiteng Zhao and Chang Ma.


## DP
```
python train.py --gpu_devices 4 5 6 7
```
## DDP
```
accelerate config
...
accelerate launch train.py --ddp_mode ddp
```
