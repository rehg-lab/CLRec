# Does Continual Learning = Catastrophic Forgetting?

In this work we investigate continual learning of reconstruction tasks and provide an insight into the difference in behavior between reconstruction tasks and classification task when learning continually. We further introduce a surprisingly simple yet effective continual learning algorithm for classification task and show the potential of using the feature representation learned in 3D shape reconstruction to serve as a proxy task for classification. In addition, we provide a visualization tool for understanding forgetting in classification task. [Link](https://arxiv.org/abs/2101.07295) to our paper and [link](https://rehg-lab.github.io/publication-pages/CLRec/) to our project webpage.

This repository consists of the code for reproducing CL of 3D shape reconstruction, proxy task and YASS results.

## Training and evaluating C-SDFNet, C-OccNet and proxy task
Follow instructions in [CL3D README](https://github.com/ngailapdi/CLRec/blob/master/CL3D/README.md)

## Training and evaluating YASS
Follow instructions in [YASS README](https://github.com/ngailapdi/CLRec/blob/master/YASS/README.md)

## Citing
```bibtex
@article{thai2021does,
  title={Does Continual Learning= Catastrophic Forgetting?},
  author={Thai, Anh and Stojanov, Stefan and Rehg, Isaac and Rehg, James M},
  journal={arXiv preprint arXiv:2101.07295},
  year={2021}
}
```


