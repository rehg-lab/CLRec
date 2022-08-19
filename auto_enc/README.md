### Environment Setup
If the environment `clpy38` is not created already, create environment using [anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
```bash
conda env create -f ../environment.yml
```
Otherwise, run the following line to activate the environment
```bash
conda activate clpy38
```

### Training autoencoder
Following is an example command to train  the autoencoder on CIFAR 100 on 2 GPUS
```bash
CUDA_VISIBLE_DEVICES=0,1 python autoenc_incr_main.py --outfile=results/autoenc_100cls_single.csv --lexp_len=500 --img_size=32 --total_classes=100 --num_iters=100 --num_epoch=250 --num_classes=1
```
