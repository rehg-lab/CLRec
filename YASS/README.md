# Incremental Object Learning from Contiguous Views
The instructions in this README follow [Incremental Object Learning from Contiguous Views](https://github.com/iolfcv/experiments/blob/master/README.md)

### Environment Setup
If the environment `sdf_net` is not created already, create environment using [anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
```bash
conda env create -f ../environment.yml
```
Otherwise, run the following line to activate the environment
```bash
conda activate sdf_net
```

### Running Incremental Learning Models

The main program has a separate train and test process. Both can run simultaneously using 1 GPU provided that batch_size + test_batch_size images can fit on GPU memory. By default, the train and test processes use the first and second GPU devices visible, unless the '--one_gpu' flag is used, in which case both use the first device visible. 

```
usage: main_incr_cifar.py [-h] [--outfile OUTFILE] [--save_all]
                          [--save_all_dir SAVE_ALL_DIR] [--resume]
                          [--resume_outfile RESUME_OUTFILE]
                          [--init_lr INIT_LR] [--init_lr_ft INIT_LR_FT]
                          [--num_epoch NUM_EPOCH]
                          [--num_epoch_ft NUM_EPOCH_FT] [--lrd LRD] [--wd WD]
                          [--batch_size BATCH_SIZE] [--llr_freq LLR_FREQ]
                          [--batch_size_test BATCH_SIZE_TEST]
                          [--lexp_len LEXP_LEN] [--size_test SIZE_TEST]
                          [--num_exemplars NUM_EXEMPLARS]
                          [--img_size IMG_SIZE]
                          [--rendered_img_size RENDERED_IMG_SIZE]
                          [--total_classes TOTAL_CLASSES]
                          [--num_classes NUM_CLASSES] [--num_iters NUM_ITERS]
                          [--algo ALGO] [--no_dist] [--pt] [--ncm] [--network]
                          [--sample SAMPLE] [--explr_neg_sig] [--random_explr]
                          [--loss LOSS] [--full_explr] [--diff_order]
                          [--subset] [--no_jitter] [--h_ch H_CH] [--s_ch S_CH]
                          [--l_ch L_CH] [--aug AUG] [--s_wo_rep]
                          [--test_freq TEST_FREQ] [--num_workers NUM_WORKERS]
                          [--one_gpu]

Incremental learning

optional arguments:
  -h, --help            show this help message and exit
  --outfile OUTFILE     Output file name (should have .csv extension)
  --save_all            Option to save models after each test_freq number of
                        learning exposures
  --save_all_dir SAVE_ALL_DIR
                        Directory to store all models in
  --resume              Resume training from checkpoint at outfile
  --resume_outfile RESUME_OUTFILE
                        Output file name after resuming
  --init_lr INIT_LR     initial learning rate
  --init_lr_ft INIT_LR_FT
                        Init learning rate for balanced finetuning (for E2E)
  --num_epoch NUM_EPOCH
                        Number of epochs
  --num_epoch_ft NUM_EPOCH_FT
                        Number of epochs for balanced finetuning (for E2E)
  --lrd LRD             Learning rate decrease factor
  --wd WD               Weight decay for SGD
  --batch_size BATCH_SIZE
                        Mini batch size for training
  --llr_freq LLR_FREQ   Learning rate lowering frequency for SGD (for E2E)
  --batch_size_test BATCH_SIZE_TEST
                        Mini batch size for testing
  --lexp_len LEXP_LEN   Number of frames in Learning Exposure
  --size_test SIZE_TEST
                        Number of test images per object
  --num_exemplars NUM_EXEMPLARS
                        number of exemplars
  --img_size IMG_SIZE   Size of images input to the network
  --rendered_img_size RENDERED_IMG_SIZE
                        Size of rendered images
  --total_classes TOTAL_CLASSES
                        Total number of classes
  --num_classes NUM_CLASSES
                        Number of classes for each learning exposure
  --num_iters NUM_ITERS
                        Total number of learning exposures (currently only
                        integer multiples of args.total_classes each class
                        seen equal number of times)
  --algo ALGO           Algorithm to run. Options : icarl, e2e, lwf
  --no_dist             Option to switch off distillation loss
  --pt                  Option to start from an ImageNet pretrained model
  --ncm                 Use nearest class mean classification (for E2E)
  --network             Use network output to classify (for iCaRL)
  --sample SAMPLE       Sampling mechanism to be performed
  --explr_neg_sig       Option to use exemplars as negative signals (for
                        iCaRL)
  --random_explr        Option for random exemplar set
  --loss LOSS           Loss to be used in classification
  --full_explr          Option to use the full exemplar set
  --diff_order          Use a random order of classes introduced
  --subset              Use a random subset of classes
  --no_jitter           Option for no color jittering (for iCaRL)
  --h_ch H_CH           Color jittering : max hue change
  --s_ch S_CH           Color jittering : max saturation change
  --l_ch L_CH           Color jittering : max lightness change
  --aug AUG             Data augmentation to perform on train data
  --s_wo_rep            Sampling train data with replacement
  --test_freq TEST_FREQ
                        Number of iterations of training after which a test is
                        done/model saved
  --num_workers NUM_WORKERS
                        Maximum number of threads spawned at anystage of
                        execution
  --one_gpu             Option to run multiprocessing on 1 GPU
```

Following is an example command to run YASS on CIFAR 100 on 2 GPUS

```bash
time CUDA_VISIBLE_DEVICES=0,1 python main_incr_cifar.py --outfile=results/test.csv --aug=e2e --batch_size_test=100 --num_exemplars=2000 --total_classes=100 --num_iters=100 --lexp_len=500 --network --sample=wg --loss=CE --random --diff_order --full_explr --no_dist --s_wo_rep
```

This project uses code based on parts of the following repository

1. [Incremental Object Learning from Contiguous Views](https://github.com/iolfcv/experiments/)

