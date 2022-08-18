### Environment Setup
The instructions in this section follow [SDFNet](https://github.com/rehg-lab/3DShapeGen/tree/master/SDFNet)

Create environment using [anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
```bash
conda env create -f ../environment.yml
```
Note that this code runs with PyTorch 1.12.1 and CUDA 10.2. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install other PyTorch and CUDA versions. `torch-scatter` might need to be reinstalled to match with the CUDA version. Please follow the instructions [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) to install `torch-scatter`.

Compile OccNet extension modules in `mesh_gen_utils`
```bash
python setup.py build_ext --inplace
```
To generate ground truths, follow [SDFNet](https://github.com/rehg-lab/3DShapeGen/tree/master/SDFNet)

### Data
1. [ShapeNetCore.v2 SDF + Point Clouds](https://www.dropbox.com/s/75lxxtmxkdr1be9/ShapeNet55_sdf.tar)
1. [3-DOF viewpoint LRBg ShapeNetCore.v2 renders](https://www.dropbox.com/s/yw03ohg04834vvv/ShapeNet55_3DOF-VC_LRBg.tar)
1. [Train/Val json on 13 classes of ShapeNetCore.v2](https://www.dropbox.com/s/7shqu6krvs9x1ib/data_split.json)
1. [Test json on 13 classes, 100 objects per class of ShapeNetCore.v2](https://www.dropbox.com/s/7ig5n662gv0uq6k/sample.json)
1. [Train/Val json on 55 classes of ShapeNetCore.v2](https://www.dropbox.com/s/7shqu6krvs9x1ib/data_split_55.json)
1. [Test json on 55 classes, 30 objects per class of ShapeNetCore.v2](https://www.dropbox.com/s/ryca8on5uhhmt04/sample_30obj_55.json)

### Training C-SDFNet
After changing the parameters in `config_shape.py` run the following to train the model from scratch
```bash
python train_shape.py
```
### Pre-trained models
The following are links to download pretrained C-SDFNet and C-OccNet models
1. [SDFNet VC with 2.5D inputs Single Exposure ShapeNetCore.v2](https://www.dropbox.com/sh/tnx34ony9y4wwsi/AABSkTG4lbtfzmLGDf6QHpOWa)
2. [OccNet VC with 2.5D inputs Single Exposure ShapeNetCore.v2](https://www.dropbox.com/sh/3jszdblnxtiit6z/AADZIvfPuTcl-wA7O1WU0UITa)
3. [SDFNet VC with 2.5D inputs Repeated Exposures ShapeNet13](https://www.dropbox.com/sh/ozdl057aiyka926/AADXpbgLBsO9Yfzw9TGOkYMYa)
4. [OccNet VC with 2.5D inputs Repeated Exposures ShapeNet13](https://www.dropbox.com/sh/eb2b0yhuq3tovqh/AABxF1A2bOgeMhpsKzYY5eUza)
5. [SDFNet OC with 2.5D inputs Repeated Exposures ShapeNet13](https://www.dropbox.com/sh/j9y8r4y6aszhb2j/AADNl6Qagd1NZ1VHIJ81hv8ea)
6. [SDFNet VC with 3D inputs Single Exposure ShapeNetCore.v2](https://www.dropbox.com/sh/wr2fctu6ldwtus8/AADZCv8ulGSHS39-6EUrybc6a?dl=0)
7. [ConvSDFNet VC with 3D inputs Single Exposure ShapeNetCore.v2](https://www.dropbox.com/sh/vmas6ja18slyap3/AABoC1ZcteY2m4VgPdAyq0xDa?dl=0)

### Testing SDFNet
```bash
python eval_shape.py
python plot_script_shape.py
```

### Evaluating Proxy Task
```bash
python main_proxy.py --num_explr=<number of exemplars, default=20>
```
This project uses code based on parts of the following repository

1. [3D Reconstruction of Novel Object Shapes from Single Images](https://github.com/rehg-lab/3DShapeGen)
