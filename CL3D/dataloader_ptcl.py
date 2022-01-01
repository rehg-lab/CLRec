import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import glob
import utils_shape as utils
import json
from torchvision import transforms

from dataloader_shape import Dataset as Dataset_Incr

class Dataset(Dataset_Incr):
    def __init__(self, config, num_points=-1,mode='train',shape_rep='occnet',coord_system='3dvc', \
        iso=0.003, perm=None, all_classes=None):
        super().__init__(config, num_points, mode, shape_rep, coord_system, iso, perm, all_classes)
        self.obj_cat_map = [(obj,cat) for cat in self.catnames \
                                for obj in self.split[cat] \
                            if os.path.exists(os.path.join(self.src_dataset_path, cat,obj)) \
                            if os.path.exists(os.path.join(self.src_ptcl_path, cat,obj))]

        self.classes = np.asarray([cat for _, cat in self.obj_cat_map])
        self.all_indices = np.arange(len(self.classes))
        self.sdf_h5_paths = [os.path.join(self.src_pt_path, cat, obj, \
                        'ori_sample.h5') \
                        for (obj, cat) in self.obj_cat_map \
                            if os.path.exists(os.path.join(self.src_ptcl_path, cat, obj))]
        self.pointcld_split_paths = [os.path.join(self.src_ptcl_path, cat, obj) \
                            for (obj, cat) in self.obj_cat_map \
                                if os.path.exists(os.path.join(self.src_ptcl_path, cat, obj))]
                                
        self.metadata_split_paths = [os.path.join(self.src_dataset_path, cat, \
                        obj, 'metadata.txt') \
                            for (obj, cat) in self.obj_cat_map \
                            if os.path.exists(os.path.join(self.src_ptcl_path, cat, obj))]
        if self.coord_system == '3dvc':
            self.hvc_metadata_split_paths = [os.path.join(self.src_dataset_path, cat, \
                        obj, '3DOF_vc_metadata.txt') \
                            for (obj, cat) in self.obj_cat_map \
                            if os.path.exists(os.path.join(self.src_ptcl_path, cat, obj))]

    def get_data_sample(self, index, img_idx=-1):
        if self.random_view:
            assert img_idx != -1

        else:
            idx = index//self.seq_len
            img_idx = index % self.seq_len

            index = idx

        index = self.current_indices[index]
        label = self.cat_map[self.classes[index]]
        return label


    def get_pointcloud_sample(self, index, img_idx=-1):
        if self.random_view:
            assert img_idx != -1
        else:
            idx = index//self.seq_len
            img_idx = index % self.seq_len

            index = idx

        index = self.current_indices[index]

        input_pointcld_path = self.pointcld_split_paths[index]
        input_pointcld_path = os.path.join(input_pointcld_path,\
                                    'pointcloud.npz')
        input_ptcld_dict = np.load(input_pointcld_path, mmap_mode='r')
        input_pointcld = input_ptcld_dict['points'].astype(np.float32)
        input_normals = input_ptcld_dict['normals'].astype(np.float32)

        if self.mode != 'test':
            input_pointcld, input_normals = utils.sample_points(input_pointcld, input_normals, self.num_points)
        else:
            sub_input_pointcld, sub_input_normals = utils.sample_points(input_pointcld, input_normals, self.num_points)

        if self.coord_system == '2dvc':

            input_metadata_path = self.metadata_split_paths[index]
            meta = np.loadtxt(input_metadata_path)
            rotate_dict = {'elev': meta[img_idx][1], 'azim': meta[img_idx][0]}

            input_pointcld = utils.apply_rotate(input_pointcld, rotate_dict)
            input_normals = utils.apply_rotate(input_normals, rotate_dict)

            if self.mode == "test":
                sub_input_pointcld = utils.apply_rotate(sub_input_pointcld, rotate_dict)
                sub_input_normals = utils.apply_rotate(sub_input_normals, rotate_dict)
                sub_input_pointcld = torch.FloatTensor(sub_input_pointcld)
                sub_input_normals = torch.FloatTensor(sub_input_normals)

        elif self.coord_system == '3dvc':
           
            input_hvc_meta_path = self.hvc_metadata_split_paths[index]
            hvc_meta = np.loadtxt(input_hvc_meta_path)
            hvc_rotate_dict = {'elev': hvc_meta[1], 'azim': hvc_meta[0]}
            input_pointcld = utils.apply_rotate(input_pointcld, hvc_rotate_dict)
            input_normals = utils.apply_rotate(input_normals, hvc_rotate_dict)


            input_metadata_path = self.metadata_split_paths[index]
            meta = np.loadtxt(input_metadata_path)
            rotate_dict = {'elev': meta[img_idx][1], 'azim': meta[img_idx][0]-180}

            input_pointcld = utils.apply_rotate(input_pointcld, rotate_dict)
            input_normals = utils.apply_rotate(input_normals, rotate_dict)

            if self.mode == 'test':
                sub_input_pointcld = utils.apply_rotate(sub_input_pointcld, hvc_rotate_dict)
                sub_input_normals = utils.apply_rotate(sub_input_normals, hvc_rotate_dict)
                sub_input_pointcld = utils.apply_rotate(sub_input_pointcld, rotate_dict)
                sub_input_normals = utils.apply_rotate(sub_input_normals, rotate_dict)
                sub_input_pointcld = torch.FloatTensor(sub_input_pointcld)
                sub_input_normals = torch.FloatTensor(sub_input_normals)
    
        input_pointcld = torch.FloatTensor(input_pointcld)
        input_normals = torch.FloatTensor(input_normals)

        if self.mode != 'test':
            return input_pointcld, input_normals

        return input_pointcld, input_normals, sub_input_pointcld, sub_input_normals


    def __getitem__(self, index):
        if self.random_view:
            img_idx = np.random.choice(self.seq_len)
        else:
            img_idx = -1
        label = self.get_data_sample(index, img_idx)

        points_data, vals_data = self.get_points_sdf_sample(index, img_idx)
        if self.shape_rep == 'occ':
            vals_data = (vals_data.cpu().numpy() <= 0.003).astype(np.float32)
            vals_data = torch.FloatTensor(vals_data)
        idx, img_idx = self.get_img_index(index, img_idx)
        if self.mode != 'test':            
            pointcloud_data, normals_data = \
                    self.get_pointcloud_sample(index, img_idx)


        if self.mode == 'test':
            pointcloud_data, normals_data, sub_pointcloud_data, sub_normals_data = self.get_pointcloud_sample(index, img_idx)
            return sub_pointcloud_data, points_data, vals_data, pointcloud_data, \
                        normals_data, self.obj_cat_map[self.current_indices[idx]], img_idx, label

        return pointcloud_data, points_data, vals_data, label


    def __len__(self):
        if len(self.current_indices) != 0:
            num_mdl = len(self.current_indices)
        else:
            num_mdl = len(self.pointcld_split_paths)
        if self.random_view:
            return num_mdl
        return num_mdl*self.seq_len



