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
from dataloader_ptcl import Dataset as Dataset_Ptc

from model_shape import SDFNet
from model_pointcloud import PointCloudNet
from model_convsdfnet import ConvSDFNet
from tqdm import tqdm
import copy

def main():
    torch.backends.cudnn.benchmark = True
    
    # log params
    log_dir = config.logging['log_dir']
    exp_name = config.logging['exp_name']
    date = datetime.now().date().strftime("%m_%d_%Y")
    log_dir = os.path.join(log_dir, exp_name, date)
    os.makedirs(log_dir, exist_ok=True)
    utils.writelogfile(config, log_dir)


    # output directory
    out_dir = config.training['out_dir']
    os.makedirs(out_dir, exist_ok=True)

    batch_size = config.training['batch_size']
    batch_size_eval = config.training['batch_size_eval']
    num_epochs = config.training['num_epochs']

    save_model_step = config.training['save_model_step']
    eval_step = config.training['eval_step']
    verbose_step = config.training['verbose_step']
    if eval_step == None:
        eval_step = int(10e9)

    num_points = config.training['num_points']

    cont = config.training['cont']

    shape_rep = config.training['shape_rep']

    coord_system = config.training['coord_system']

    # Number of repeated exposures. num_rep = 1 for single exposure case
    num_rep = config.training['num_rep']
    # Num class per exposure
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
        train_dataset = Dataset(config, num_points=num_points, mode='train', \
            shape_rep=shape_rep, coord_system=coord_system)
        eval_train_dataset = Dataset(config, mode='val', shape_rep=shape_rep, \
            coord_system=coord_system)
        val_dataset = Dataset(config, mode='val', shape_rep=shape_rep, coord_system=coord_system)
    else:
        train_dataset = Dataset_Ptc(config, num_points=num_points, mode='train', \
            shape_rep=shape_rep, coord_system=coord_system)
        eval_train_dataset = Dataset_Ptc(config, mode='val', shape_rep=shape_rep, \
            coord_system=coord_system)
        val_dataset = Dataset_Ptc(config, mode='val', shape_rep=shape_rep, coord_system=coord_system)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=12, shuffle=True, pin_memory=True)
    eval_train_loader = torch.utils.data.DataLoader(
        eval_train_dataset, batch_size=batch_size_eval, num_workers=12,\
        drop_last=False, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size_eval, num_workers=12,\
        drop_last=False, pin_memory=True)

    # Get all training classes
    all_classes_orig = train_dataset.catnames
    total_classes = len(all_classes_orig)

    cat_map = {}
    perm_path = './perm/rep_%s_%s_%s.npz'%(str(num_rep), str(nclass), str(total_classes))
    ###### Note: fix perm for odd number of classes
    if not os.path.exists(perm_path):
        perm_all = np.concatenate([np.random.permutation(np.arange(len(all_classes_orig))) \
            for _ in range(num_rep)])
        # Reshape to N exposures x nclass
        perm_all = perm_all.reshape((len(perm_all)//nclass,nclass))
        perm_all = np.random.permutation(perm_all)
        all_classes = np.asarray(all_classes_orig)[perm_all]
    else:
        all_classes = np.load(perm_path)\
            ['all_classes']
    
    for cl_ind, cl_group in enumerate(all_classes):
        for sub_cl_ind, cl in enumerate(cl_group):
            if cl not in cat_map:
                cat_map[cl] = len(cat_map.keys())

    train_dataset.update_class_map(cat_map)
    eval_train_dataset.update_class_map(cat_map)
    val_dataset.update_class_map(cat_map)

    # Model
    print('Initializing network...')
    if model_type == "SDFNet":
        model = SDFNet(config)
        model_eval = SDFNet(config)
    elif model_type == "PointCloudNet":
        model = PointCloudNet(config)
        model_eval = PointCloudNet(config)
    elif model_type == "ConvSDFNet":
        model = ConvSDFNet(config)
        model_eval = ConvSDFNet(config)
    else:
        raise Exception("Model type not supported")

    # Initialize training
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    if shape_rep == 'occ':
        criterion = nn.BCEWithLogitsLoss()
    elif shape_rep == 'sdf' :
        criterion = utils.LpLoss
    else:
        raise Exception('Algorithm not supported')
        
    metric_val_array = []
    epoch_val_array = []
    loss_val_array = []

    # Stores val IoU for each class at each learning exposure
    # num exposure x num_classes
    metric_val_matrr = np.zeros((len(all_classes),len(all_classes_orig)),dtype=np.float32)
    seen_classes = []

    if shape_rep == 'occ':
        max_metric_train = 0
    elif shape_rep == 'sdf':
        max_metric_train = np.zeros(2, dtype=np.float32)
    metric_train_array = []
    epoch_train_array = []
    loss_train_array = []

    current_counter = 0


    if cont is not None:
        try:
            current_counter = int(cont.split('-')[1])+1
        except Exception:
            print("Current counter is not an integer")
        checkpoint = torch.load(os.path.join(out_dir, cont))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        perm_load = []
        seen_classes_load = []
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        epoch_it = checkpoint['epoch']
        if epoch_it != num_epochs:
            raise Exception("Please make sure to continue from the last epoch of a learning exposure")
        if os.path.exists(os.path.join(out_dir, 'train.npz')):
            # Load saved data
            try:
                train_npz = np.load(os.path.join(out_dir, 'train.npz'), allow_pickle=True)
                metric_train_array = train_npz['metric']
                epoch_train_array = train_npz['epoch']
                loss_train_array = train_npz['loss']
                perm_load = train_npz['perm']
                seen_classes_load = train_npz['seen_classes']
                # If num_epochs = 500, verbose_step = 10 and current_counter = 5
                # then ckpt = 250 
                ckpt = ((num_epochs-verbose_step)//verbose_step+1)*current_counter
                metric_train_array = list(metric_train_array[:ckpt])
                loss_train_array = list(loss_train_array[:ckpt])
                epoch_train_array = list(epoch_train_array[:ckpt])
            except Exception:
                print('Cannot load train npz')
        if os.path.exists(os.path.join(out_dir, 'val.npz')):
            try:
                val_npz = np.load(os.path.join(out_dir, 'val.npz'), allow_pickle=True)
                if len(perm_load) == 0:
                    perm_load = val_npz['perm']

                    seen_classes_load = val_npz['seen_classes']

                metric_val_array = val_npz['metric']
                epoch_val_array = val_npz['epoch']
                loss_val_array = val_npz['loss']

                ckpt = ((num_epochs-eval_step)//eval_step+1)*current_counter
                metric_val_array = list(metric_val_array[:ckpt])
                loss_val_array = list(loss_val_array[:ckpt])
                epoch_val_array = list(epoch_val_array[:ckpt])

                metric_val_matrr_load = val_npz['metric_matrr']
                metric_val_matrr[:metric_val_matrr_load.shape[0],:metric_val_matrr_load.shape[1]]\
                    = metric_val_matrr_load
            except Exception:
                print('Cannot load val npz')

        if not os.path.exists(perm_path) \
            and len(perm_load) != 0 and len(seen_classes_load) != 0:
            all_classes = perm_load
            seen_classes = seen_classes_load
        elif os.path.exists(perm_path) \
            and len(perm_load) != 0 and len(seen_classes_load) != 0:
            seen_classes = seen_classes_load
        
        seen_classes = list(seen_classes)


    # Saving meta config for loading
    meta_config_path = os.path.join(out_dir, 'meta_config.npz')
    np.savez(meta_config_path, training=config.training, \
        testing=config.testing, data_setting=config.data_setting, \
        logging=config.logging, path=config.path)
    if not os.path.exists(perm_path):
        os.makedirs('./perm',exist_ok=True)
        np.savez(perm_path, all_classes=all_classes)


    # Data parallel
    model = torch.nn.DataParallel(model).cuda()
    model_eval = torch.nn.DataParallel(model_eval).cuda()


    print(metric_val_matrr.shape)
    print('Start training...')
    for cl_count, cl_group in enumerate(all_classes[current_counter:]):
        cl_count += current_counter

        for cl in cl_group:
            print('Class: ', cl)

            if cl not in seen_classes:
                seen_classes.append(cl)

            # Get current classes for train and val
            train_dataset.get_current_data_class(cl)
            eval_train_dataset.get_current_data_class(cl)

        epoch_it = 0
        if shape_rep == 'occ':
            max_metric_train = 0
        elif shape_rep == 'sdf':
            max_metric_train = np.zeros(2, dtype=np.float32)

        while True:
            epoch_it += 1
            if num_epochs is not None and epoch_it > num_epochs:
                break
            print("Starting epoch %s"%(epoch_it))
            model = train(model, criterion, optimizer, train_loader, \
                batch_size, epoch_it, shape_rep)

            if epoch_it % save_model_step == 0:
                print("Saving model...")
                torch.save({
                    'epoch': epoch_it,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},\
                        os.path.join(out_dir, 'model-%s-%s.pth.tar'%(cl_count,epoch_it)))

            if epoch_it % verbose_step == 0:
                print("Evaluating on train data...")
                mean_loss, mean_metric = eval(model, criterion, optimizer, \
                    eval_train_loader, batch_size, epoch_it, shape_rep)
                print('Mean loss on train set: %.4f'%(mean_loss))
                if shape_rep == 'occ':

                    print('Mean IoU on train set: %.4f'%(mean_metric[0]))
                elif shape_rep == 'sdf':
                    print('Mean IoU on train set: %.4f'%(mean_metric[2]))
                    print('Mean accuracy on train set: %.4f'%(mean_metric[1]))

                metric_train_array.append(mean_metric)
                epoch_train_array.append(epoch_it)
                loss_train_array.append(mean_loss)

                np.savez(os.path.join(out_dir, 'train.npz'), metric=metric_train_array, \
                    epoch=epoch_train_array, loss=loss_train_array, perm=all_classes, \
                    seen_classes=seen_classes, current_counter=cl_count)

                # Saving best model based on metric
                if shape_rep == 'occ':
                    if mean_metric[0] > max_metric_train:
                        print('Saving best model')
                        max_metric_train = copy.deepcopy(mean_metric[0])
                        print('Saving best model')
                        torch.save(model.module.state_dict(), \
                            os.path.join(out_dir, 'best_model_train-%s.pth.tar'%(cl_count)))
                        model_eval.module.load_state_dict(\
                            copy.deepcopy(model.module.state_dict()))
                elif shape_rep == 'sdf':
                    if mean_metric[2] > max_metric_train[0]:
                        print('Saving best model')
                        max_metric_train[0] = copy.deepcopy(mean_metric[2])
                        torch.save(model.module.state_dict(), \
                            os.path.join(out_dir, 'best_model_iou_train-%s.pth.tar'%(cl_count)))
                        model_eval.module.load_state_dict(\
                            copy.deepcopy(model.module.state_dict()))
                                                
                    if mean_metric[1] > max_metric_train[1]:
                        max_metric_train[1] = copy.deepcopy(mean_metric[1])
                        torch.save(model.module.state_dict(), os.path.join(out_dir, 'best_model_acc_train-%s.pth.tar'%(cl_count)))

            if epoch_it % eval_step == 0:
                for s in range(len(seen_classes)):
                    val_dataset.clear()
                    val_dataset.get_current_data_class(seen_classes[s])

                    print('Evaluating on test data of class %s...'%(seen_classes[s]))
                    mean_loss_val, mean_metric_val = eval(\
                        model_eval, criterion, optimizer, val_loader, \
                        batch_size, epoch_it, shape_rep)
                    print('Mean loss on val set: %.4f'%(mean_loss_val))
                    if shape_rep == 'occ':
                        metric_val_matrr[cl_count, s] = mean_metric_val[0]
                        print('Mean IoU on val set: %.4f'%(mean_metric_val[0]))
                    elif shape_rep == 'sdf':
                        metric_val_matrr[cl_count, s] = mean_metric_val[2]
                        print('Mean IoU on val set: %.4f'%(mean_metric_val[2]))
                        print('Mean accuracy on val set: %.4f'%(mean_metric_val[1]))

                    metric_val_array.append(mean_metric_val)
                    epoch_val_array.append(epoch_it)
                    loss_val_array.append(mean_loss_val)

                print(metric_val_matrr)
                np.savez(os.path.join(out_dir, 'val.npz'), metric=metric_val_array, \
                        epoch=epoch_val_array, loss=loss_val_array, metric_matrr=metric_val_matrr, \
                        perm=all_classes, seen_classes=seen_classes, current_counter=cl_count)

                del mean_loss_val
        
        train_dataset.clear()
        eval_train_dataset.clear()

def train(model, criterion, optimizer, train_loader, \
            batch_size, epoch_it, shape_rep):
    model.train()
    with tqdm(total=int(len(train_loader)), ascii=True) as pbar:
        for mbatch in train_loader:
            img_input, points_input, values, labels = mbatch
            img_input = Variable(img_input).cuda()

            points_input = Variable(points_input).cuda()
            values = Variable(values).cuda()

            labels = Variable(labels).cuda()


            optimizer.zero_grad()
            
            logits = model(points_input, img_input)
            if shape_rep == 'occ':
                loss = criterion(logits, values)
            elif shape_rep == 'sdf':
                loss = criterion(logits, values)

            loss.backward()
            optimizer.step()

            del loss
            
            pbar.update(1)
    return model


def eval(model, criterion, optimizer, loader, batch_size, epoch_it, shape_rep):
    model.eval()
    loss_collect = []
    metric_collect = []
    if shape_rep == 'occ':
        sigmoid = torch.nn.Sigmoid()

    with tqdm(total=int(len(loader)), ascii=True) as pbar:
        with torch.no_grad():
            for mbatch in loader:
                img_input, points_input, values, labels = mbatch
                img_input = Variable(img_input).cuda()

                points_input = Variable(points_input).cuda()
                values = Variable(values).cuda()

                labels = Variable(labels).cuda()


                optimizer.zero_grad()

                logits = model(points_input, img_input)

                loss = criterion(logits, values)

                loss_collect.append(loss.data.cpu().item())

                if shape_rep == 'occ':
                    logits = sigmoid(logits)

                    iou = utils.compute_iou(logits.detach().cpu().numpy(), \
                                values.detach().cpu().numpy())
                    metric_collect.append([iou])
                elif shape_rep == 'sdf':
                    acc_sign, acc_thres, iou = utils.compute_acc(\
                                        logits.detach().cpu().numpy(), \
                                        values.detach().cpu().numpy())
                    metric_collect.append([acc_sign, acc_thres, iou])

                pbar.update(1)

    mean_loss = np.mean(np.array(loss_collect))
    if shape_rep == 'occ':
        mean_metric = np.mean(np.concatenate(metric_collect))
        mean_metric = [mean_metric]
    else:
        mean_metric = np.mean(np.array(metric_collect), axis=0)

    return mean_loss, mean_metric

if __name__ == '__main__':
    main()
