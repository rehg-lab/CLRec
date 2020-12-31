### Environment Setup
Please see the instructions in [SDFNet](https://github.com/rehg-lab/3DShapeGen/tree/master/SDFNet)

### Data
1. [ShapeNetCore.v2 SDF + Point Clouds](https://www.dropbox.com/s/75lxxtmxkdr1be9/ShapeNet55_sdf.tar)
1. [3-DOF viewpoint LRBg ShapeNetCore.v2 renders](https://www.dropbox.com/s/yw03ohg04834vvv/ShapeNet55_3DOF-VC_LRBg.tar)
1. [Train/Val/Test json on 13 classes of ShapeNetCore.v2](https://www.dropbox.com/s/7shqu6krvs9x1ib/data_split.json)
1. [Train/Val/Test json on 55 classes of ShapeNetCore.v2](https://www.dropbox.com/s/7shqu6krvs9x1ib/data_split_55.json)

### Training SDFNet
After changing the parameters in `config.py` run the following to train the model from scratch
```bash
python train_shape.py
```
### Pre-trained SDFNet
The following are links to download pretrained SDFNet models
1. [C-SDFNet VC Single Exposure ShapeNetCore.v2](https://www.dropbox.com/s/p6pxqyxk1p5gp8f/best_model_gt_dn_3DOF.pth.tar)
2. [C-OccNet VC Single Exposure ShapeNetCore.v2](https://www.dropbox.com/s/uavq47qt80ltbyq/best_model_pred_dn_3DOF.pth.tar)
3. [C-SDFNet VC Repeated Exposures ShapeNet13](https://www.dropbox.com/s/uavq47qt80ltbyq/best_model_pred_dn_3DOF.pth.tar)
4. [C-OccNet VC Repeated Exposures ShapeNet13](https://www.dropbox.com/s/uavq47qt80ltbyq/best_model_pred_dn_3DOF.pth.tar)
5. [C-SDFNet OC Repeated Exposures ShapeNet13](https://www.dropbox.com/s/uavq47qt80ltbyq/best_model_pred_dn_3DOF.pth.tar)
### Testing SDFNet
```bash
python eval_shape.py
python plot_script.py

This project uses code based on parts of the following repositories

1. [3D Reconstruction of Novel Object Shapes from Single Images](3D Reconstruction of Novel Object Shapes from Single Images)