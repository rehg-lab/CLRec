import torch
import numpy as np
import os
import trimesh
from tqdm import tqdm
import config_shape as config
from dataloader_shape import Dataset
from dataloader_ptcl import Dataset as Dataset_Ptc

from model_shape import SDFNet
from model_pointcloud import PointCloudNet
from model_convsdfnet import ConvSDFNet
from torch.autograd import Variable
import torch.optim as optim
import utils_shape as utils


def main():
    out_dir = config.training['out_dir']
    shape_rep = config.training['shape_rep']
    cont = config.training['cont']

    eval_task_name = config.testing['eval_task_name']
    eval_dir = os.path.join(out_dir, 'eval')
    eval_task_dir = os.path.join(eval_dir, eval_task_name)
    os.makedirs(eval_task_dir, exist_ok=True)

    batch_size_test = config.testing['batch_size_test']
    coord_system = config.training['coord_system']

    box_size = config.testing['box_size']

    split_counter = config.testing['split_counter']+1

    nclass = config.training['nclass']

    # Whether to use pointclouds as input
    pointcloud = config.training['pointcloud']

    # Get model
    model_type = config.training['model']
    if model_type == None:
        model_type = 'SDFNet' #default to be SDFNet

    # Dataset
    print('Loading data...')
    if not pointcloud:
        test_dataset = Dataset(config, mode='test', shape_rep=shape_rep, \
            coord_system=coord_system)
    else:
        test_dataset = Dataset_Ptc(config, mode='test', shape_rep=shape_rep, coord_system=coord_system)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size_test, num_workers=12,pin_memory=True)
    all_classes_orig = test_dataset.catnames

    # Load info
    val_file = os.path.join(out_dir, 'train.npz')
    val_file = np.load(val_file, allow_pickle=True)
    all_classes = val_file['perm']

    cat_map = {}

    for cl_ind, cl_group in enumerate(all_classes):
        for sub_cl_ind, cl in enumerate(cl_group):
            if cl not in cat_map:
                cat_map[cl] = len(cat_map.keys())

    test_dataset.update_class_map(cat_map)
    current_counter = 0
    if not cont is None:
       try:
           current_counter = int(cont.split('-')[1])+1
       except Exception:
           print('Current counter is not an integer')

    # Loading model
    if model_type == "SDFNet":
        model = SDFNet(config)
    elif model_type == "PointCloudNet":
        model = PointCloudNet(config)
    elif model_type == "ConvSDFNet":
        model = ConvSDFNet(config)
    else:
        raise Exception("Model type not supported")
    model = torch.nn.DataParallel(model).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)


    out_obj_cat_all = []
    out_pose_all = []
    out_cd_all = []
    out_normals_all = []
    out_iou_all = []
    out_fscore_all = []
    out_acc_all = []

    seen_classes = list(set(all_classes[:current_counter].reshape(-1)))
    print('Start counter: ', current_counter)
    print('End counter: ', split_counter-1)
    for cl_count, cl_group in enumerate(all_classes[current_counter:split_counter]):
        print('Num seen classes: ', len(seen_classes))
        cl_count += current_counter
        # Load model and reload loader
        if shape_rep == 'sdf':
            model_path = 'best_model_iou_train-%s.pth.tar'%(cl_count)
        elif shape_rep == 'occ':
            model_path = 'best_model_train-%s.pth.tar'%(cl_count)

        new_classes = []
        for cl in cl_group:
            if cl not in seen_classes:
                seen_classes.append(cl)
                new_classes.append(cl)
        model_path = os.path.join(out_dir, model_path)
        model.module.load_state_dict(torch.load(model_path))
        model.eval()

        out_obj_cat_cl = []
        out_pose_cl = []
        out_cd_cl = []
        out_normals_cl = []
        out_iou_cl = []
        out_fscore_cl = []
        out_acc_cl = []

        for s in range(len(seen_classes)):
            print('Evaluating exposure %s, class %s'\
                %(cl_count, seen_classes[s]))

            out_obj_cat = []
            out_pose = []
            out_cd = []
            out_normals = []
            out_iou = []
            out_fscore = []
            out_acc = []
            test_dataset.clear()
            test_dataset.get_current_data_class(seen_classes[s])

            with tqdm(total=int(len(test_loader)), ascii=True) as pbar:
                with torch.no_grad():
                    for mbatch in test_loader:
                        img_input, points_input, values, pointclouds, normals, \
                            obj_cat, pose, labels = mbatch
                        img_input = Variable(img_input).cuda()

                        points_input = Variable(points_input).cuda()
                        values = Variable(values).cuda()
                        labels = Variable(labels).cuda()

                        optimizer.zero_grad()

                        obj, cat = obj_cat
                        cat_path = os.path.join(eval_task_dir, cat[0])

                        os.makedirs(cat_path, exist_ok=True)
                        if shape_rep == 'occ':
                            mesh = utils.generate_mesh(img_input, points_input, \
                                model.module)
                            obj_path = os.path.join(cat_path, '%s.obj' % obj[0])
                            mesh.export(obj_path)
                        elif shape_rep == 'sdf':
                            obj_path = os.path.join(cat_path, '%s-%s.obj' \
                                % (cl_count, obj[0]))
                            sdf_path = os.path.join(cat_path, '%s-%s.dist' \
                                % (cl_count, obj[0]))
                            mesh = utils.generate_mesh_mise_sdf(img_input, \
                                points_input, model.module, box_size=box_size,\
                                upsampling_steps=2, resolution=64)
                            mesh.export(obj_path)

                        # Save gen info
                        out_obj_cat.append(obj_cat)
                        out_pose.append(pose)
                        
                        # Calculate metrics
                        if shape_rep == 'occ':
                            out_dict = utils.eval_mesh(mesh, pointclouds, normals,\
                                        points_input, values)
                        elif shape_rep == 'sdf':
                            # load the mesh
                            if os.path.exists(obj_path):
                                #### Load mesh
                                try:
                                    mesh = trimesh.load(obj_path)
                                except Exception:
                                    mesh = None
                            else:
                                mesh = None
                            sdf_val = model(points_input, img_input)

                            out_dict = utils.eval_mesh(mesh, pointclouds, normals, \
                                        points_input, values, shape_rep='sdf',\
                                        sdf_val=sdf_val)
                        
                        out_cd.append(out_dict['cd'])
                        out_normals.append(out_dict['normals'])
                        out_iou.append(out_dict['iou'])
                        out_fscore.append(out_dict['fscore'])
                        pbar.update(1)

            out_obj_cat_cl.append(out_obj_cat)
            out_pose_cl.append(out_pose)
            out_cd_cl.append(out_cd)
            out_normals_cl.append(out_normals)
            out_iou_cl.append(out_iou)
            out_fscore_cl.append(out_fscore)

        out_obj_cat_all.append(out_obj_cat_cl)
        out_pose_all.append(out_pose_cl)
        out_cd_all.append(out_cd_cl)
        out_normals_all.append(out_normals_cl)
        out_iou_all.append(out_iou_cl)
        out_fscore_all.append(out_fscore_cl)
        np.savez(os.path.join(eval_task_dir, 'out-%s.npz'%(split_counter)), \
            obj_cat=np.array(out_obj_cat_all), pose=np.array(out_pose_all),\
            cd=np.array(out_cd_all), normals=np.array(out_normals_all),\
            iou=np.array(out_iou_all), fscore=np.array(out_fscore_all),\
            all_classes=all_classes, seen_classes=seen_classes)                  

if __name__ == '__main__':
    main()



