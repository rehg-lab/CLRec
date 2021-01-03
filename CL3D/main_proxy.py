########### Train a classifier on top of pretrained shape features
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import os
import sys
import torch.multiprocessing as mp
import subprocess
import config_shape as config
from datetime import datetime
import utils_shape as utils
from dataloader_shape import Dataset

from model_shape import SDFNet
from tqdm import tqdm
import copy
import argparse

parser = argparse.ArgumentParser(description="Proxy Task")
parser.add_argument("--num_explr", default=20, type=int,
                    help="Number of exemplars")

def calc_acc(plabels, glabels):
    '''
    Calculates classification accuracy
    args:
        plabels: predicted labels
        glabels: ground truth labels
    '''
    mean_acc = 0
    label_set = set(glabels)

    per_class_acc = {}
    for gl in label_set:
        pl = plabels[glabels == gl]
        pl_pred = pl[pl == gl]
        mean_acc += len(pl_pred)/len(pl)
        per_class_acc[gl] = len(pl_pred)/len(pl)
    return mean_acc/len(label_set), per_class_acc

def forward_pass(model, loader, train_loader, num_classes, mode='val'):
    model.eval()
    feats = []
    exemplar_feats, exemplar_labels = get_exemplar_feats(model, train_loader)
    glabels = []
    with tqdm(total=int(len(loader)), ascii=True) as pbar:
        with torch.no_grad():
            for data in loader:
                if mode == 'val':
                    img_input, points_input, values, labels = data
                else:
                    img_input, points_input, values, _, _, _, _, labels = data
                img_input = Variable(img_input).cuda()

                feats.append(model(img_input).cpu().numpy())
                glabels.append(labels)
                pbar.update(1)
    feats = np.concatenate(feats, axis=0)
    glabels = np.concatenate(glabels)

    dist_matr = compute_dist(feats, exemplar_feats)
    plabels = np.argmax(dist_matr,axis=1)
    return calc_acc(plabels, glabels)

def get_exemplar_feats(model, loader):
    '''
    Get exemplar features and labels 
    args:
        model: shape model
        loader: train loader
    '''
    model.eval()
    glabels = []
    feats = []
    with tqdm(total=int(len(loader)), ascii=True) as pbar:
        with torch.no_grad():
            for data in loader:
                img_input, points_input, values, labels = data
                img_input = Variable(img_input).cuda()

                feats.append(model(img_input).cpu().numpy())
                glabels.append(labels)
                pbar.update(1)
    feats = np.concatenate(feats, axis=0)
    glabels = np.concatenate(glabels)
    mean_feats = []
    mean_labels = []
    # Gets mean features for each ground truth label
    for g in set(glabels):
        mfeats = np.mean(feats[glabels == g],axis=0)
        mean_feats.append(mfeats.reshape(1,-1))
        mean_labels.append(g)
    feats = np.concatenate(mean_feats,axis=0)
    glabels = np.array(mean_labels)
    glabels_argsort = np.argsort(glabels)
    feats = feats[glabels_argsort]
    glabels = glabels[glabels_argsort]
    return feats, glabels

def compute_dist(feats, exemplar_feats):
    '''
    Gets cosine distances between exemplar features and test features
    args:
        feats: test features
        exemplar_feats: exemplar features
    '''
    normalized_feats = feats/np.linalg.norm(feats,axis=1,keepdims=True)
    normalized_ex_feats = exemplar_feats/np.linalg.norm(exemplar_feats,axis=1,keepdims=True)
    dist = np.dot(normalized_feats,normalized_ex_feats.transpose(1,0))
    return dist

def main():
    args = parser.parse_args()
    num_explr = args.num_explr

    torch.backends.cudnn.benchmark=True
    out_dir = config.training['out_dir']
    os.makedirs(out_dir, exist_ok=True)

    num_classes = config.training['nclass']
    coord_system = config.training['coord_system']
    shape_rep = config.training['shape_rep']

    # Dataset
    print('Loading data...')
    train_dataset = Dataset(num_points=2048, mode='train', shape_rep=shape_rep, \
        coord_system=coord_system, config=config)
    test_dataset = Dataset(num_points=2048, mode='test', shape_rep=shape_rep, \
        coord_system=coord_system, config=config)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=256, num_workers=12, shuffle=True,\
            pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=100, num_workers=12, drop_last=False,pin_memory=True)

    # Model
    print('Initializing network...')
    shape_model = SDFNet(config=config)

    shape_train_file = os.path.join(out_dir, 'train.npz')
    if os.path.exists(shape_train_file):
        shape_train_file = np.load(shape_train_file, allow_pickle=True)
    else:
        raise Exception("Train npz for shape model does not exist")
    all_classes = shape_train_file['perm']

    cat_map = {}

    for cl_ind, cl_group in enumerate(all_classes):
        for sub_cl_ind, cl in enumerate(cl_group):
            if cl not in cat_map:
                cat_map[cl] = len(cat_map.keys())
    train_dataset.update_class_map(cat_map)
    test_dataset.update_class_map(cat_map)

    train_dataset.init_exemplar()
    seen_classes = []

    test_accs = []
    classifier_dir = os.path.join(out_dir, 'self_sup_classifier')
    os.makedirs(classifier_dir, exist_ok=True)
    for cl_count, cl_group in enumerate(all_classes):
        for cl in cl_group:
            if cl not in seen_classes:
                test_dataset.get_current_data_class(cl)
                seen_classes.append(cl)

            train_dataset.get_current_data_class(cl)
            train_dataset.sample_exemplar_rep(cl, num_explr)

        train_dataset.set_train_on_exemplar()

        model_path = 'best_model_iou_train-%s.pth.tar'%(cl_count)

        model_path = os.path.join(out_dir, model_path)
        shape_model.load_state_dict(torch.load(model_path))
        model = torch.nn.DataParallel(shape_model.encoder).cuda()
        test_acc, per_class_acc = forward_pass(model, test_loader, train_loader, \
            len(seen_classes), mode='test')

        test_accs.append(test_acc)
        train_dataset.clear()

        print('Accuracy on test set: ', test_acc)

        np.savez(os.path.join(classifier_dir, 'val.npz'), test_acc=test_accs)

if __name__ == '__main__':
    main()