import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import glob
import utils
import json
from torchvision import transforms

class Dataset(Dataset):
    def __init__(self, config, num_points=-1,mode='train',shape_rep='sdf',\
        coord_system='3dvc', iso=0.003, perm=None, all_classes=None):

        self.mode = mode
        self.coord_system = coord_system

        self.config = config
        
        self.input_size = self.config.data_setting['input_size']
        self.num_points = num_points

        self.src_dataset_path = self.config.path['src_dataset_path']
        self.input_image_path = self.config.path['input_image_path']
        self.input_depth_path = self.config.path['input_depth_path']
        self.input_normal_path = self.config.path['input_normal_path']
        self.input_seg_path = self.config.path['input_seg_path']

        self.img_extension = self.config.data_setting['img_extension']


        self.src_pt_path = self.config.path['src_pt_path']
        self.data_split_json_path = self.config.path['data_split_json_path']

        self.shape_rep = shape_rep

        self.categories = self.config.data_setting['categories']

        self.iso = iso

        with open(self.data_split_json_path, 'r') as data_split_file:
            self.data_splits = json.load(data_split_file)
        self.split = self.data_splits[self.mode]
        self.random_view = self.config.data_setting['random_view']
        self.seq_len = self.config.data_setting['seq_len']

        self.catnames = sorted(list(self.split.keys()))
        self.cat_map = {}

        if self.categories is not None:
            self.catnames = sorted([c for c in self.catnames \
                            if c in self.categories])

        self.obj_cat_map = [(obj,cat) for cat in self.catnames \
                                for obj in self.split[cat] \
                            if os.path.exists(os.path.join(self.src_dataset_path, cat,obj))]

        self.classes = np.asarray([cat for _, cat in self.obj_cat_map])
        self.all_indices = np.arange(len(self.classes))
        # Init current training indices
        self.current_indices = []

        # Get all image paths
        self.img_paths = [os.path.join(self.src_dataset_path, cat, obj) \
                                for (obj, cat) in self.obj_cat_map \
                                if os.path.exists(os.path.join(self.src_dataset_path, cat,obj))]
        # Get all sdf paths
        self.sdf_h5_paths = [os.path.join(self.src_pt_path, cat, obj, 'ori_sample.h5') \
                        for (obj, cat) in self.obj_cat_map \
                        if os.path.exists(os.path.join(self.src_dataset_path, cat, obj))]
        # Get all pointcloud paths
        self.pointcld_split_paths = [os.path.join(self.src_pt_path, cat, obj) \
                            for (obj, cat) in self.obj_cat_map \
                            if os.path.exists(os.path.join(self.src_dataset_path, cat, obj))]
        # Get all metadata paths                        
        self.metadata_split_paths = [os.path.join(self.src_dataset_path, cat, \
                        obj, 'metadata.txt') \
                            for (obj, cat) in self.obj_cat_map \
                            if os.path.exists(os.path.join(self.src_dataset_path, cat, obj))]
        if self.coord_system == '3dvc':
            # self.hvc_metadata_split_paths = [os.path.join(self.src_dataset_path, cat, \
            #             obj, 'hard_vc_metadata.txt') \
            #                 for (obj, cat) in self.obj_cat_map \
            #                 if os.path.exists(os.path.join(self.src_dataset_path, cat, obj))]
            self.hvc_metadata_split_paths = [os.path.join(self.src_dataset_path, cat, \
                        obj, '3DOF_vc_metadata.txt') \
                            for (obj, cat) in self.obj_cat_map \
                            if os.path.exists(os.path.join(self.src_dataset_path, cat, obj))]
        # Input RGB image transform function
        self.img_transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.image_split_paths, self.depth_split_paths, \
            self.normal_split_paths, self.seg_split_paths = self.load_image_paths()

    def load_image_paths(self):
        '''
        Loads all images paths
        '''
        if self.input_image_path is not None:
            image_split_files = [sorted(glob.glob(
                        os.path.join(obj_path, self.input_image_path, '*.%s' 
                            % self.img_extension)))[:] \
                                for obj_path in self.img_paths \
                                if os.path.exists(\
                                    os.path.join(obj_path, self.input_image_path))]
        else:
            image_split_files = None

        if self.input_depth_path is not None:
            seg_split_files = [sorted(glob.glob(
                    os.path.join(obj_path, self.input_seg_path, '*.%s'%self.img_extension\
                            )))[:] for obj_path in self.img_paths \
                                    if os.path.exists(\
                                        os.path.join(obj_path, self.input_seg_path))]

            depth_split_files = [sorted(glob.glob(
                    os.path.join(obj_path, self.input_depth_path, '*.npz'\
                            )))[:] for obj_path in self.img_paths \
                                    if os.path.exists(\
                                        os.path.join(obj_path, self.input_depth_path))]
        else:
            depth_split_files = None
            seg_split_files = None
        if self.input_normal_path is not None:
            normal_split_files = [sorted(glob.glob(
                    os.path.join(obj_path, self.input_normal_path, '*.%s' 
                        % self.img_extension)))[:] \
                            for obj_path in self.img_paths \
                            if os.path.exists(\
                                os.path.join(obj_path, self.input_normal_path))]
        else:
            normal_split_files = None


        return image_split_files, depth_split_files, normal_split_files, seg_split_files

    def get_data_sample(self, index, img_idx=-1):
        '''
        Get each image sample for __getitem__
        '''
        if self.random_view:
            assert img_idx != -1
        else:
            # Object index
            idx = index//self.seq_len
            # Image/pose index
            img_idx = index % self.seq_len

            index = idx        

        index = self.current_indices[index]

        if self.image_split_paths is not None:

            input_image = self.image_split_paths[index][img_idx]
            image_data = Image.open(input_image).convert('RGB')
            image_data = self.img_transform(image_data)
            image_data = np.array(image_data.numpy())
        else:
            image_data = np.array([])

        if self.depth_split_paths is not None:
            input_depth = self.depth_split_paths[index][img_idx]
            input_seg = self.seg_split_paths[index][img_idx]
            depth_data = np.load(input_depth)['img']
            depth_min_max = np.load(input_depth)['min_max']
            min_d, max_d = depth_min_max[0], depth_min_max[1]

            # Resize depth npz from 256x256 to 224x224
            depth_image = Image.fromarray(np.uint8(depth_data*255.))
            depth_image = depth_image.resize(\
                (self.input_size, self.input_size))            
            depth_data = np.array(depth_image)/255.

            # Convert depth to range min to max
            depth_data = 1 - depth_data
            seg_data = Image.open(input_seg).convert('L')
            seg_data = seg_data.resize((self.input_size, self.input_size))
            seg_data = np.array(seg_data)/255. # 0-1 with object as 1

            depth_data[seg_data == 0.] = 10. # Set background to max value

            depth_data[seg_data != 0.] = depth_data[seg_data != 0.]*(max_d-min_d)+min_d
            depth_data = np.expand_dims(depth_data, axis=2)
            depth_data = depth_data.transpose(2,0,1)

            if len(image_data) == 0:
                image_data = depth_data
            else:
                image_data = np.concatenate(\
                    [image_data, depth_data],axis=0)

        else:
            depth_data = None
        if self.normal_split_paths is not None:
            input_normal = self.normal_split_paths[index][img_idx]

            normal_data = Image.open(input_normal).convert('RGB')
            # Resize and normalize
            normal_data = normal_data.resize(\
                (self.input_size, self.input_size))
            normal_data = np.array(normal_data)/255.
            normal_data = normal_data.transpose(2,0,1)
            if len(image_data) == 0:
                image_data = normal_data
            else: 
                image_data = np.concatenate(\
                    [image_data, normal_data],axis=0)
        else:
            normal_data = None

        image_data = torch.FloatTensor(image_data)
        label = self.cat_map[self.classes[index]]

        return image_data, label

    def get_points_sdf_sample(self, index, img_idx=-1):
        '''
        Get point sample for __getitem__
        '''
        assert self.sdf_h5_paths is not None
        if self.random_view:
            assert img_idx != -1
        else:
            idx = index//self.seq_len
            img_idx = index % self.seq_len

            index = idx

        index = self.current_indices[index]

        ori_pt, ori_sdf_val, input_points, input_sdfs, norm_params, \
            sdf_params  = utils.get_sdf_h5(self.sdf_h5_paths[index])
        input_points, input_sdfs = utils.sample_points(input_points,\
                                           input_sdfs,\
                                           self.num_points)

        if self.coord_system == '2dvc':

            input_metadata_path = self.metadata_split_paths[index]
            meta = np.loadtxt(input_metadata_path)
            rotate_dict = {'elev': meta[img_idx][1], 'azim': meta[img_idx][0]}

            input_points = utils.apply_rotate(input_points, rotate_dict)

        elif self.coord_system == '3dvc':
            # Rotate points for 3-DOF VC
            input_hvc_meta_path = self.hvc_metadata_split_paths[index]
            hvc_meta = np.loadtxt(input_hvc_meta_path)
            hvc_rotate_dict = {'elev': hvc_meta[1], 'azim': hvc_meta[0]}
            input_points = utils.apply_rotate(input_points, hvc_rotate_dict)

            input_metadata_path = self.metadata_split_paths[index]
            meta = np.loadtxt(input_metadata_path)
            rotate_dict = {'elev': meta[img_idx][1], 'azim': meta[img_idx][0]-180}

            input_points = utils.apply_rotate(input_points, rotate_dict)

        input_points = torch.FloatTensor(input_points)
        input_sdfs = torch.FloatTensor(input_sdfs)
        return input_points, input_sdfs


    def get_pointcloud_sample(self, index, img_idx=-1):
        '''
        Get pointcloud sample for __getitem__ during testing
        '''
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

        if self.coord_system == '2dvc':

            input_metadata_path = self.metadata_split_paths[index]
            meta = np.loadtxt(input_metadata_path)
            rotate_dict = {'elev': meta[img_idx][1], 'azim': meta[img_idx][0]}

            input_pointcld = utils.apply_rotate(input_pointcld, rotate_dict)
            input_normals = utils.apply_rotate(input_normals, rotate_dict)

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

        input_pointcld = torch.FloatTensor(input_pointcld)
        input_normals = torch.FloatTensor(input_normals)

        return input_pointcld, input_normals

    def __getitem__(self, index):
        if self.random_view:
            img_idx = np.random.choice(self.seq_len)
        else:
            img_idx = -1
        image_data, label = self.get_data_sample(index, img_idx)

        points_data, vals_data = self.get_points_sdf_sample(index, img_idx)
        if self.shape_rep == 'occ':
            vals_data = (vals_data.cpu().numpy() <= self.iso).astype(np.float32)
            vals_data = torch.FloatTensor(vals_data)

        if self.mode == 'test':
            idx, img_idx = self.get_img_index(index, img_idx)            
            pointcloud_data, normals_data = \
                    self.get_pointcloud_sample(index, img_idx)
            return image_data, points_data, vals_data, pointcloud_data, \
                        normals_data, self.obj_cat_map[self.current_indices[idx]], \
                        img_idx, label
        return image_data, points_data, vals_data, label

    def __len__(self):
        if len(self.current_indices) != 0:
            num_mdl = len(self.current_indices)
        else:
            if self.image_split_paths != None:
                num_mdl = len(self.image_split_paths)
            elif self.depth_split_paths != None:
                num_mdl = len(self.depth_split_paths)
            elif self.normal_split_paths != None:
                num_mdl = len(self.normal_split_paths)
            else:
                raise Exception("Must have at least 1 input image type")
        if self.random_view:
            return num_mdl
        return num_mdl*self.seq_len

    def get_img_index(self, index, img_idx):
        if img_idx == -1:
            idx = index//self.seq_len
            img_idx = index % self.seq_len
        else:
            idx = index

        return idx, img_idx

    def clear(self):
        self.current_indices = []

    def get_current_data_class(self, cls):
        '''
        Get data from specified class
        args:
            cls: ground truth class
        '''
        indices = self.current_indices
        self.current_indices = self.all_indices[self.classes == cls]
        # Append distinct classes
        if len(indices) != 0:
            self.current_indices = np.concatenate([indices, self.current_indices])

    def update_class_map(self, cat_map):
        '''
        Init class map
        '''
        self.cat_map = cat_map

    def init_exemplar(self):
        '''
        Init exemplar
        '''
        self.exemplar_indices = []

    def remove_exemplar(self, r_cls, m):
        '''
        Remove exemplar randomly
        '''
        o_cls = self.cat_map[r_cls]
        cls_indices = self.exemplar_indices[o_cls]
        if len(cls_indices) > m:
            keep_samples_ind = np.random.choice(len(cls_indices),m,replace=False)
            keep_samples = cls_indices[keep_samples_ind]
            self.exemplar_indices[o_cls] = keep_samples

    def sample_exemplar(self, r_cls, m):
        current_cls_ind = self.classes[self.current_indices]
        current_cls_sample_ind = self.current_indices[current_cls_ind == r_cls]
        chosen_samples_ind = np.random.choice(len(current_cls_sample_ind),m,replace=False)
        self.exemplar_indices.append(current_cls_sample_ind[chosen_samples_ind])

    def sample_exemplar_rep(self, r_cls, m):
        '''
        Sample exemplar randomly
        '''
        n_classes_seen = len(self.exemplar_indices)
        o_cls = self.cat_map[r_cls]
        current_cls_ind = self.classes[self.current_indices]
        current_cls_sample_ind = self.current_indices[current_cls_ind == r_cls]
        chosen_samples_ind = np.random.choice(len(current_cls_sample_ind),m,replace=False)

        if o_cls < n_classes_seen:
            self.exemplar_indices[o_cls] = current_cls_sample_ind[chosen_samples_ind]
        else:
            self.exemplar_indices.append(current_cls_sample_ind[chosen_samples_ind])

    def set_train_on_exemplar(self):
        self.current_indices = np.concatenate(self.exemplar_indices, axis=0)