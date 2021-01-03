# Does Continual Learning = Catastrophic Forgetting?

In this work we investigate continual learning of reconstruction tasks and provide an insight into the difference in behavior between reconstruction tasks and classification task when learning continually. We further introduce a surprisingly simple yet effective continual learning algorithm for classification task and show the potential of using the feature representation learned in 3D shape reconstruction to serve as a proxy task for classification. In addition, we provide a visualization tool for understanding forgetting in classification task. [Link]() to our paper and [link]() to our project webpage.

This repository consists of the code for reproducing CL of 3D shape reconstruction, proxy task and YASS results.

## Training and evaluating C-SDFNet, C-OccNet and proxy task
Follow instructions in [CL3D README](https://github.com/ngailapdi/CLRec/blob/master/CL3D/README.md)

## Training and evaluating YASS
Follow instructions in [YASS README](https://github.com/ngailapdi/CLRec/blob/master/YASS/README.md)

## Citing
```bibtex
@misc{TO-ADD,
      title={Does Continual Learning = Catastrophic Forgetting?}, 
      author={Anh Thai and Stefan Stojanov and Isaac Rehg and James M. Rehg},
      year={2021},
      eprint={to-add},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


