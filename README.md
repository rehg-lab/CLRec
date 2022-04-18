# The Surprising Positive Knowledge Transfer in Continual 3D Object Shape Reconstruction

In this work we investigate continual learning of reconstruction tasks which surprisingly do not suffer from catastrophic forgetting and exhibit positive forward knowledge transfer. In addition, we provide a novel analysis of knowledge transfer ability in CL. We further show the potential of using the feature representation learned in 3D shape reconstruction to serve as a proxy task for classification. [Link](https://arxiv.org/abs/2101.07295) to our paper and [link](https://rehg-lab.github.io/publication-pages/CLRec/) to our project webpage.

This repository consists of the code for reproducing CL of 3D shape reconstruction, proxy task and autoencoder results.

## Training and evaluating Single Object 3D Shape Reconstruction and proxy task
Follow instructions in [CL3D README](https://github.com/ngailapdi/CLRec/blob/master/CL3D/README.md)

## Training and evaluating autoencoder
Follow instructions in [Autoencoder README](https://github.com/ngailapdi/CLRec/blob/master/auto_enc/README.md)

## Citing
```bibtex
@misc{thai2021surprising,
    title={The Surprising Positive Knowledge Transfer in Continual 3D Object Shape Reconstruction},
    author={Anh Thai and Stefan Stojanov and Zixuan Huang and Isaac Rehg and James M. Rehg},
    year={2021},
    eprint={2101.07295},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```


