import numpy as np
import os
from datetime import datetime
import copy
import torch
from mesh_gen_utils.libmise import MISE
from mesh_gen_utils.libmesh import check_mesh_contains
from mesh_gen_utils import libmcubes
import trimesh
from mesh_gen_utils.libkdtree import KDTree
from torch.autograd import Variable
import h5py
import torch.nn as nn
from PIL import Image


def writelogfile(config, log_dir):
    log_file_name = os.path.join(log_dir, 'log.txt')
    with open(log_file_name, "a+") as log_file:
        log_string = get_log_string(config)
        log_file.write(log_string)

def get_log_string(config):
    now = str(datetime.now().strftime("%H:%M %d-%m-%Y"))
    log_string = ""
    log_string += " -------- Hyperparameters and settings -------- \n"
    log_string += "{:25} {}\n".format('Time:', now)
    log_string += "{:25} {}\n".format('Mini-batch size:', \
        config.training['batch_size'])
    log_string += "{:25} {}\n".format('Batch size eval:', \
        config.training['batch_size_eval'])
    log_string += "{:25} {}\n".format('Num epochs:', \
        config.training['num_epochs'])
    log_string += "{:25} {}\n".format('Out directory:', \
        config.training['out_dir'])
    log_string += "{:25} {}\n".format('Random view:', \
        config.data_setting['random_view'])
    log_string += "{:25} {}\n".format('Sequence length:', \
        config.data_setting['seq_len'])
    log_string += "{:25} {}\n".format('Input size:', \
        config.data_setting['input_size'])
    log_string += " -------- Data paths -------- \n"
    log_string += "{:25} {}\n".format('Dataset path', \
        config.path['src_dataset_path'])
    log_string += "{:25} {}\n".format('Point path', \
        config.path['src_pt_path'])
    log_string += " ------------------------------------------------------ \n"
    return log_string

def compute_iou(occ1, occ2):
    ''' Computes the Intersection over Union (IoU) value for two sets of
    occupancy values.

    Args:
        occ1 (tensor): first set of occupancy values
        occ2 (tensor): second set of occupancy values
    '''
    occ1 = np.asarray(occ1)
    occ2 = np.asarray(occ2)

    # Put all data in second dimension
    # Also works for 1-dimensional data
    if occ1.ndim >= 2:
        occ1 = occ1.reshape(occ1.shape[0], -1)
    if occ2.ndim >= 2:
        occ2 = occ2.reshape(occ2.shape[0], -1)

    # Convert to boolean values
    occ1_temp = copy.deepcopy(occ1)
    occ2_temp = copy.deepcopy(occ2)
    occ1 = (occ1 >= 0.5)
    occ2 = (occ2 >= 0.5)

    # Compute IOU
    area_union = (occ1 | occ2).astype(np.float32).sum(axis=-1)
    if (area_union == 0).any():
        return 0.

    area_intersect = (occ1 & occ2).astype(np.float32).sum(axis=-1)

    iou = (area_intersect / area_union)
    if isinstance(iou, (list,np.ndarray)):
        iou = np.mean(iou, axis=0)
    return iou

def compute_acc(sdf_pred, sdf, thres=0.01, iso=0.003):
    # import pdb; pdb.set_trace()
    sdf_pred = np.asarray(sdf_pred)
    sdf = np.asarray(sdf)

    acc_sign = (((sdf_pred-iso) * (sdf-iso)) > 0).mean(axis=-1)
    acc_sign = np.mean(acc_sign, axis=0)

    occ_pred = sdf_pred <= iso
    occ = sdf <= iso

    iou = compute_iou(occ_pred, occ)

    acc_thres = (np.abs(sdf_pred-sdf) <= thres).mean(axis=-1)
    acc_thres = np.mean(acc_thres, axis=0)
    return acc_sign, acc_thres, iou

def get_sdf_h5(sdf_h5_file):
    h5_f = h5py.File(sdf_h5_file, 'r')
    try:
        if ('pc_sdf_original' in h5_f.keys()
                and 'pc_sdf_sample' in h5_f.keys()
                and 'norm_params' in h5_f.keys()):
            ori_sdf = h5_f['pc_sdf_original'][:].astype(np.float32)
            sample_sdf = h5_f['pc_sdf_sample'][:].astype(np.float32)
            ori_pt = ori_sdf[:,:3]#, ori_sdf[:,3]
            ori_sdf_val = None
            if sample_sdf.shape[1] == 4:
                sample_pt, sample_sdf_val = sample_sdf[:,:3], sample_sdf[:,3]
            else:
                sample_pt, sample_sdf_val = None, sample_sdf[:, 0]
            norm_params = h5_f['norm_params'][:]
            sdf_params = h5_f['sdf_params'][:]
        else:
            raise Exception("no sdf and sample")
    finally:
        h5_f.close()
    return ori_pt, ori_sdf_val, sample_pt, sample_sdf_val, norm_params, sdf_params

def apply_rotate(input_points, rotate_dict):
    theta_azim = rotate_dict['azim']
    theta_elev = rotate_dict['elev']
    theta_azim = np.pi+theta_azim/180*np.pi
    theta_elev = theta_elev/180*np.pi
    r_elev = np.array([[1,       0,          0],
                        [0, np.cos(theta_elev), -np.sin(theta_elev)],
                        [0, np.sin(theta_elev), np.cos(theta_elev)]])
    r_azim = np.array([[np.cos(theta_azim), 0, np.sin(theta_azim)],
                        [0,               1,       0],
                        [-np.sin(theta_azim),0, np.cos(theta_azim)]])

    rotated_points = r_elev@r_azim@input_points.T
    return rotated_points.T

def sample_points(input_points, input_occs, num_points):
    if num_points != -1:
        idx = torch.randint(len(input_points), size=(num_points,))
    else:
        idx = torch.arange(len(input_points))
    selected_points = input_points[idx, :]
    selected_occs = input_occs[idx]
    return selected_points, selected_occs

def normalize_imagenet(x):
    ''' Normalize input images according to ImageNet standards.
    Args:
        x (tensor): input images
    '''
    x = x.clone()
    x[:, 0] = (x[:, 0] - 0.485) / 0.229
    x[:, 1] = (x[:, 1] - 0.456) / 0.224
    x[:, 2] = (x[:, 2] - 0.406) / 0.225
    return x

def LpLoss(logits, sdf, p=1, thres=0.01, weight=4.):

    sdf = Variable(sdf.data, requires_grad=False).cuda()
    loss = torch.abs(logits-sdf).pow(p).cuda()
    weight_mask = torch.ones(loss.shape).cuda()
    weight_mask[torch.abs(sdf) < thres] =\
             weight_mask[torch.abs(sdf) < thres]*weight 
    loss = loss * weight_mask
    loss = torch.sum(loss, dim=-1, keepdim=False)
    loss = torch.mean(loss)
    return loss

def generate_mesh(img, points, model, threshold=0.2, box_size=1.7, \
            resolution0=16, upsampling_steps=2):
    '''
    Generate mesh function for OccNet
    '''
    model.eval()

    threshold = np.log(threshold) - np.log(1. - threshold)
    mesh_extractor = MISE(
        resolution0, upsampling_steps, threshold)
    p = mesh_extractor.query()

    with torch.no_grad():
        feats = model.encoder(img)

    while p.shape[0] != 0:
        pq = torch.FloatTensor(p).cuda()
        pq = pq / mesh_extractor.resolution

        pq = box_size * (pq - 0.5)

        with torch.no_grad():
            pq = pq.unsqueeze(0)
            occ_pred = model.decoder(pq, feats)
        values = occ_pred.squeeze(0).detach().cpu().numpy()
        values = values.astype(np.float64)
        mesh_extractor.update(p, values)

        p = mesh_extractor.query()
    value_grid = mesh_extractor.to_dense()

    mesh = extract_mesh(value_grid, feats, box_size, threshold)
    return mesh

def extract_mesh(value_grid, feats, box_size, threshold, constant_values=-1e6):
    n_x, n_y, n_z = value_grid.shape
    value_grid_padded = np.pad(
            value_grid, 1, 'constant', constant_values=constant_values)
    vertices, triangles = libmcubes.marching_cubes(
            value_grid_padded, threshold)
    # Shift back vertices by 0.5
    vertices -= 0.5
    # Undo padding
    vertices -= 1
    # Normalize
    vertices /= np.array([n_x-1, n_y-1, n_z-1])
    vertices = box_size * (vertices - 0.5)

    # Create mesh
    mesh = trimesh.Trimesh(vertices, triangles, process=False)

    return mesh

def eval_mesh(mesh, pointcloud_gt, normals_gt, points, val_gt, \
                num_fscore_thres=6, n_points=300000, shape_rep='occ', \
                sdf_val=None, iso=0.003):

    if mesh is not None and type(mesh)==trimesh.base.Trimesh and len(mesh.vertices) != 0 and len(mesh.faces) != 0:
        pointcloud, idx = mesh.sample(n_points, return_index=True)
        pointcloud = pointcloud.astype(np.float32)
        normals = mesh.face_normals[idx]
    else:
        if shape_rep == 'occ':
            return {'iou': 0., 'cd': 2*np.sqrt(3), 'completeness': np.sqrt(3),\
                    'accuracy': np.sqrt(3), 'normals_completeness': -1,\
                    'normals_accuracy': -1, 'normals': -1, \
                    'fscore': np.zeros(6, dtype=np.float32), \
                    'precision': np.zeros(6, dtype=np.float32), \
                    'recall': np.zeros(6, dtype=np.float32)}
        return {'iou': [0.,0.], 'cd': 2*np.sqrt(3), 'completeness': np.sqrt(3),\
                    'accuracy': np.sqrt(3), 'normals_completeness': -1,\
                    'normals_accuracy': -1, 'normals': -1, \
                    'fscore': np.zeros(6, dtype=np.float32), \
                    'precision': np.zeros(6, dtype=np.float32), \
                    'recall': np.zeros(6, dtype=np.float32)}
    # Eval pointcloud
    pointcloud = np.asarray(pointcloud)
    pointcloud_gt = np.asarray(pointcloud_gt.squeeze(0))
    normals = np.asarray(normals)
    normals_gt = np.asarray(normals_gt.squeeze(0))

    ####### Normalize
    pointcloud /= (2*np.max(np.abs(pointcloud)))
    pointcloud_gt /= (2*np.max(np.abs(pointcloud_gt)))

    # Completeness: how far are the points of the target point cloud
    # from thre predicted point cloud
    completeness, normals_completeness = distance_p2p(
            pointcloud_gt, normals_gt, pointcloud, normals)

    # Accuracy: how far are th points of the predicted pointcloud
    # from the target pointcloud
    accuracy, normals_accuracy = distance_p2p(
        pointcloud, normals, pointcloud_gt, normals_gt
    )

    # Get fscore
    fscore_array, precision_array, recall_array = [], [], []
    for i, thres in enumerate([0.5, 1, 2, 5, 10, 20]):
        fscore, precision, recall = calculate_fscore(\
            accuracy, completeness, thres/100.)
        fscore_array.append(fscore)
        precision_array.append(precision)
        recall_array.append(recall)
    fscore_array = np.array(fscore_array, dtype=np.float32)
    precision_array = np.array(precision_array, dtype=np.float32)
    recall_array = np.array(recall_array, dtype=np.float32)

    accuracy = accuracy.mean()
    normals_accuracy = normals_accuracy.mean()

    completeness = completeness.mean()
    normals_completeness = normals_completeness.mean()

    cd = completeness + accuracy
    normals = 0.5*(normals_completeness+normals_accuracy)

    # Compute IoU
    if shape_rep == 'occ':
        occ_mesh = check_mesh_contains(mesh, points.cpu().numpy().squeeze(0))
        iou = compute_iou(occ_mesh, val_gt.cpu().numpy().squeeze(0))
    else:
        occ_mesh = check_mesh_contains(mesh, points.cpu().numpy().squeeze(0))
        val_gt_np = val_gt.cpu().numpy()
        occ_gt = val_gt_np <= iso
        iou = compute_iou(occ_mesh, occ_gt) 

        # sdf iou
        sdf_iou, _, _ = compute_acc(sdf_val.cpu().numpy(),\
                        val_gt.cpu().numpy()) 
        iou = np.array([iou, sdf_iou])

    return {'iou': iou, 'cd': cd, 'completeness': completeness,\
                'accuracy': accuracy, \
                'normals_completeness': normals_completeness,\
                'normals_accuracy': normals_accuracy, 'normals': normals, \
                'fscore': fscore_array, 'precision': precision_array,\
                'recall': recall_array}

def calculate_fscore(accuracy, completeness, threshold):
    recall = np.sum(completeness < threshold)/len(completeness)
    precision = np.sum(accuracy < threshold)/len(accuracy)
    if precision + recall > 0:
        fscore = 2*recall*precision/(recall+precision)
    else:
        fscore = 0
    return fscore, precision, recall


def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    ''' Computes minimal distances of each point in points_src to points_tgt.

    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    kdtree = KDTree(points_tgt)
    dist, idx = kdtree.query(points_src)

    if normals_src is not None and normals_tgt is not None:
        normals_src = \
            normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = \
            normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to mehtod not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array(
            [np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product
    

def generate_mesh_mise_sdf(img, points, model, threshold=0.003, box_size=1.7, \
            resolution=64, upsampling_steps=2):
    '''
    Generates mesh for sdf representations using MISE algorithm
    '''
    model.eval()

    resolution0 = resolution // (2**upsampling_steps)

    total_points = (resolution+1)**3
    split_size = int(np.ceil(total_points*1.0/128**3))
    mesh_extractor = MISE(
        resolution0, upsampling_steps, threshold)
    p = mesh_extractor.query()
    with torch.no_grad():
        feats = model.encoder(img)
    while p.shape[0] != 0:

        pq = p / mesh_extractor.resolution

        pq = box_size * (pq - 0.5)
        occ_pred = []
        with torch.no_grad():
            if pq.shape[0] > 128**3:

                pq = np.array_split(pq, split_size)

                for ind in range(split_size):

                    occ_pred_split = model.decoder(torch.FloatTensor(pq[ind])\
                            .cuda().unsqueeze(0), feats)
                    occ_pred.append(occ_pred_split.cpu().numpy().reshape(-1))
                occ_pred = np.concatenate(np.asarray(occ_pred),axis=0)
                values = occ_pred.reshape(-1)
            else:
                pq = torch.FloatTensor(pq).cuda().unsqueeze(0)
                occ_pred = model.decoder(pq, feats)
                values = occ_pred.squeeze(0).detach().cpu().numpy()
        values = values.astype(np.float64)
        mesh_extractor.update(p, values)

        p = mesh_extractor.query()
    value_grid = mesh_extractor.to_dense()
    mesh = extract_mesh(value_grid, feats, box_size, threshold, constant_values=1e6)
    return mesh










