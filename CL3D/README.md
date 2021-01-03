### Environment Setup
The instructions in this section follow [SDFNet](https://github.com/rehg-lab/3DShapeGen/tree/master/SDFNet)

Create environment using [anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
```bash
conda env create -f ../environment.yml
```
Compile OccNet extension modules in `mesh_gen_utils`
```bash
python setup.py build_ext --inplace
```
To generate ground truths and perform testing, change the path in `isosurface/LIB_PATH` to your miniconda/anaconda libraries, for example
```bash
export LD_LIBRARY_PATH="<path_to_anaconda>/lib:<path_to_anaconda>/envs/sdf_net/lib:./isosurface:$LD_LIBRARY_PATH" 
source isosurface/LIB_PATH
```

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
1. [C-SDFNet VC Single Exposure ShapeNetCore.v2](https://www.dropbox.com/sh/tnx34ony9y4wwsi/AABSkTG4lbtfzmLGDf6QHpOWa)
2. [C-OccNet VC Single Exposure ShapeNetCore.v2](https://www.dropbox.com/sh/3jszdblnxtiit6z/AADZIvfPuTcl-wA7O1WU0UITa)
3. [C-SDFNet VC Repeated Exposures ShapeNet13](https://www.dropbox.com/sh/ozdl057aiyka926/AADXpbgLBsO9Yfzw9TGOkYMYa)
4. [C-OccNet VC Repeated Exposures ShapeNet13](https://www.dropbox.com/sh/eb2b0yhuq3tovqh/AABxF1A2bOgeMhpsKzYY5eUza)
5. [C-SDFNet OC Repeated Exposures ShapeNet13](https://www.dropbox.com/sh/j9y8r4y6aszhb2j/AADNl6Qagd1NZ1VHIJ81hv8ea)

### Testing C-SDFNet
```bash
python eval_shape.py
python plot_script.py
```

### Evaluating Proxy Task
```bash
python main_proxy.py --num_explr=<# exemplars, default=20>
```
This project uses code based on parts of the following repository

1. [3D Reconstruction of Novel Object Shapes from Single Images](https://github.com/rehg-lab/3DShapeGen)