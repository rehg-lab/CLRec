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
import utils
from dataloader_shape import Dataset

from model_shape import SDFNet
from classifier_model import SDFNetClassifier
from tqdm import tqdm
import copy


torch.backends.cudnn.benchmark=True

# model_path = '/data/DevLearning/SDFNet_model_output/3dof_paper/best_model_iou.pth.tar'
# model_path = '/data/DevLearning/SDFNet_model_output/incr_13_hvc_rep_5/best_model_iou_train-129.pth.tar'
out_dir = config.training['out_dir']
os.makedirs(out_dir, exist_ok=True)
num_epoch = 100
# num_classes = 13
# mode = 'train'
# ft_fc = True

num_classes = 5
mode = 'train'
# ft_fc = True


class Classifier(nn.Module):
    def __init__(self, model, num_feat=256):
        super().__init__()
        self.n_classes = 0
        self.num_feat = num_feat
        self.feat_extract = model
        self.fc = nn.Linear(num_feat, 1)

    def increment_class(self, new_classes):
        import pdb; pdb.set_trace()

        weight = self.fc.weight.data
        out_features = self.fc.out_features

        if self.n_classes == 0:
            new_out_features = len(new_classes)
        else:
            new_out_features = out_features + len(new_classes)
        self.fc = nn.Linear(self.num_feat, new_out_features)

        self.fc.weight.data[:out_features] = weight
        self.n_classes += len(new_classes)

    def forward(self, img):
        feat = self.feat_extract(img)
        logits = self.fc(feat)
        return logits

# Dataset
print('Loading data...')
if mode == 'train':
    train_dataset = Dataset(num_points=2048, mode='train', algo='sdf', \
        coord_system='hvc', config=config)
    # val_dataset = Dataset(mode='val', rep=rep, coord_system=coord_system)
    val_dataset = Dataset(num_points=2048, mode='val', algo='sdf', \
        coord_system='hvc', config=config)
test_dataset = Dataset(num_points=2048, mode='test', algo='sdf', coord_system='hvc', config=config)

if mode == 'train':
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=256, num_workers=12, shuffle=True,\
            pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=100, num_workers=12, \
            drop_last=False,pin_memory=True)


test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=100, num_workers=12, drop_last=False,pin_memory=True)

# Model
print('Initializing network...')
shape_model = OccupancyNetwork(model=None, config=config)
model = Classifier(model=shape_model.encoder)
# model = SDFNetClassifier(model=shape_model)

############ Load model
# model_shape.load_state_dict(torch.load(model_path))
#########################

# model_classifier = Classifier(model_shape, enc=True, num_classes=num_classes)
# model_classifier.feat_extract = model_shape.encoder
# for p in model_classifier.feat_extract.parameters():
#     p.requires_grad=False
# model_classifier = torch.nn.DataParallel(model_classifier).cuda()


# Initialize training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)



val_file = os.path.join(out_dir, 'val.npz')
val_file = np.load(val_file, allow_pickle=True)
all_classes = val_file['perm']

# all_classes = sorted(np.array(['02691156','02828884','02933112','02958343','03001627','03211117','03636649','03691459','04090263','04256520','04379243','04401088','04530566']))
# all_classes = np.asarray(all_classes)
# all_classes = all_classes.reshape(-1,1)

cat_map = {}

for cl_ind, cl_group in enumerate(all_classes):
    for sub_cl_ind, cl in enumerate(cl_group):
        if cl not in cat_map:
            cat_map[cl] = len(cat_map.keys())
if mode == 'train':
    train_dataset.update_class_map(cat_map)
    val_dataset.update_class_map(cat_map)
test_dataset.update_class_map(cat_map)

# import pdb; pdb.set_trace()

def train(model, epoch, loader, optimizer):
    model.train()
    train_loss = 0.0
    with tqdm(total=len(loader)) as pbar:
        for i, data in enumerate(loader):
            # import pdb; pdb.set_trace()
            img_input, points_input, values, labels = data
            img_input = Variable(img_input).cuda()
            labels = Variable(labels).cuda()
            points_input = Variable(points_input).cuda()
            optimizer.zero_grad()
            # sdf, logits = model(points_input, img_input)
            logits = model(img_input)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*len(labels)
            if (i+1) % 1 == 0:
                tqdm.write("Epoch %d, Iter [%d/%d] Loss: %.4f" 
                    %(epoch, i+1, 
                      len(train_dataset)//256, loss.item()))
            pbar.update(1)
            # del sdf
        train_loss /= len(train_dataset)

        train_epoch_loss.append(train_loss)

def eval(model, loader, optimizer, mode=None):
    model.eval()
    acc = 0.0
    total_samples = 0
    with tqdm(total=int(len(loader)), ascii=True) as pbar:
        with torch.no_grad():
            for data in loader:
                if mode is None:
                    img_input, points_input, values, labels = data
                else:
                    img_input, points_input, values, pointclouds, normals, \
                        obj_cat, pose, labels = data
                img_input = Variable(img_input).cuda()
                labels = Variable(labels).cuda()

                points_input = Variable(points_input).cuda()

                optimizer.zero_grad()

                # sdf, logits = model(points_input, img_input)
                logits = model(img_input)


                loss = criterion(logits, labels)

                pred_labels = torch.argmax(logits, dim=1)
                acc += torch.sum(pred_labels == labels).item()
                total_samples += len(labels)

                pbar.update(1)
    acc = acc/total_samples
    return acc

def clustering(features, num_clusters):
    from scipy.cluster.vq import vq, kmeans2, whiten
    # np.random.seed((1000,1000))
    w_feat = whiten(features)
    centroids, pseudo_labels = kmeans2(w_feat, num_clusters)
    return centroids, pseudo_labels

def label_mapping_naive(plabels, glabels):
    label_set = set(glabels)
    label_map = {}
    for gl in list(label_set):
        pl = plabels[glabels == gl]
        counts = np.bincount(pl)
        label_map[gl] = np.argmax(counts)
    return label_map

def label_mapping(centroids, features, glabels):
    label_set = set(glabels)
    dist_queue = []
    label_map = {}

    for i, gl in enumerate(label_set):
        fets = features[glabels == gl]
        mean_feats = np.mean(fets, axis=0)
        mean_feats = mean_feats/np.linalg.norm(mean_feats)
        dists = [1-np.dot(mean_feats,c/np.linalg.norm(c)) for c in centroids]
        dist_queue.append(dists)
    dist_queue = np.concatenate(dist_queue)
    as_dist_queue = np.argsort(dist_queue)
    cluster_set = copy.deepcopy(label_set)

    for ind in as_dist_queue:
        gt_i = ind//len(centroids)
        c_i = ind%len(centroids)
        if gt_i in label_set and c_i in cluster_set:
            label_map[gt_i] = c_i
            cluster_set.remove(c_i)
            label_set.remove(gt_i)
    return label_map


def calc_acc(plabels, glabels, label_map):
    mean_acc = 0
    label_set = set(glabels)

    per_class_acc = {}
    for gl in label_set:
        mapped_label = label_map[gl]
        pl = plabels[glabels == gl]
        pl_pred = pl[pl == mapped_label]
        mean_acc += len(pl_pred)/len(pl)
        per_class_acc[gl] = len(pl_pred)/len(pl)
    return mean_acc/len(label_set), per_class_acc

def forward_pass(model, loader, train_loader, num_classes, mode='val'):
    model.eval()
    feats = []
    exemplar_feats, exemplar_labels = get_exemplar_feats(model, train_loader)
    # import pdb; pdb.set_trace()

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
    label_map = {i:i for i in range(num_classes)}

    # import pdb; pdb.set_trace()
    # centroids, plabels = clustering(feats, num_classes)
    # label_map = label_mapping(centroids, feats, glabels)

    print(label_map)
    return calc_acc(plabels, glabels, label_map)

def get_exemplar_feats(model, loader):
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
    # import pdb; pdb.set_trace()

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
    normalized_feats = feats/np.linalg.norm(feats,axis=1,keepdims=True)
    normalized_ex_feats = exemplar_feats/np.linalg.norm(exemplar_feats,axis=1,keepdims=True)
    dist = np.dot(normalized_feats,normalized_ex_feats.transpose(1,0))
    return dist


'''
seen_classes = []
model = torch.nn.DataParallel(model).cuda()
test_accs = []

for cl_count, cl_group in enumerate(all_classes):
    best_acc = 0.0
    best_epoch = 0

    test_acc = 0.0
    train_epoch_loss = []
    if mode == 'train':
        model_path = 'best_model_iou_train-%s.pth.tar'%(cl_count)
    else:
        classifier_dir = os.path.join(out_dir, 'classifier')
        os.makedirs(classifier_dir, exist_ok=True)
        model_path = 'best_model_classifier_acc-%s.pth.tar'%(cl_count)
        model_path = os.path.join(classifier_dir, model_path)
        if not os.path.exists(model_path):
            test_accs.append(0.0)
            continue

    new_classes = []
    for cl in cl_group:
        if mode == 'train' and not ft_fc:
            train_dataset.get_current_data_class(cl)
            val_dataset.get_current_data_class(cl)
        if cl not in seen_classes:
            seen_classes.append(cl)
            new_classes.append(cl)

            #################### FT-FC
            if mode == 'train' and ft_fc:
                train_dataset.get_current_data_class(cl)
                val_dataset.get_current_data_class(cl)
            test_dataset.get_current_data_class(cl)

    import pdb; pdb.set_trace()
    model.module.increment_class(new_classes)
    model = torch.nn.DataParallel(model.module).cuda()
    # import pdb; pdb.set_trace()

    model_path = os.path.join(out_dir, model_path)
    if mode == 'train':
        # model.module.rec_model.load_state_dict(torch.load(model_path))
        # for p in model.module.rec_model.parameters():
        #     p.requires_grad=False

        shape_model.load_state_dict(torch.load(model_path))
        import pdb; pdb.set_trace()

        for p in model.module.feat_extract.parameters():
            p.requires_grad=False
    else:
        checkpoint = torch.load(model_path)
        model.module.load_state_dict(checkpoint['state_dict'])
        # import pdb; pdb.set_trace()


    if mode == 'train':
        for epoch in range(num_epoch):
            train(model, epoch, \
                train_loader, optimizer)
            import pdb; pdb.set_trace()
            acc = eval(model, val_loader, optimizer)

            print('Accuracy on val set: ', acc)

            classifier_dir = os.path.join(out_dir, 'classifier')
            os.makedirs(classifier_dir, exist_ok=True)

            model_save_path = 'best_model_classifier_acc_ft_fc-%s.pth.tar'%(cl_count)

            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
            # import pdb; pdb.set_trace()

                test_acc = eval(model, test_loader, optimizer,mode='test')
                print('Accuracy on test set: ', test_acc)

            torch.save({"epoch": epoch+1, "state_dict": model.module.state_dict(), \
                            "best_acc": best_acc, "best_epoch": best_epoch, \
                            "optimizer": optimizer.state_dict()}, \
                            os.path.join(classifier_dir, model_save_path))



            np.savez(os.path.join(classifier_dir,'val_ft_fc-%s.npz'%(cl_count)), best_acc=best_acc, \
                best_epoch=best_epoch, train_epoch_loss=train_epoch_loss, test_acc=test_acc)
        if not ft_fc:
            train_dataset.clear()
            val_dataset.clear()
    else:

        test_acc = eval(model, test_loader, optimizer, mode='test')
        test_accs.append(test_acc)
        print('Accuracy on test set: ', test_acc)
        np.savez(os.path.join(classifier_dir, 'test.npz'), test_acc=test_accs)

'''
train_dataset.init_exemplar()
seen_classes = []
# model = torch.nn.DataParallel(shape_model.encoder).cuda()
test_accs = []
classifier_dir = os.path.join(out_dir, 'self_sup_classifier')
os.makedirs(classifier_dir, exist_ok=True)
for cl_count, cl_group in enumerate(all_classes):
    for cl in cl_group:
        if cl not in seen_classes:
            test_dataset.get_current_data_class(cl)

            seen_classes.append(cl)
        train_dataset.get_current_data_class(cl)
        # import pdb; pdb.set_trace()
        train_dataset.sample_exemplar_rep(cl, 20)
    train_dataset.set_train_on_exemplar()
    # import pdb; pdb.set_trace()

    model_path = 'best_model_iou_train-%s.pth.tar'%(cl_count)

    model_path = os.path.join(out_dir, model_path)
    shape_model.load_state_dict(torch.load(model_path))
    model = torch.nn.DataParallel(shape_model.encoder).cuda()
    test_acc, per_class_acc = forward_pass(model, test_loader, train_loader, len(seen_classes), mode='test')

    test_accs.append(test_acc)
    train_dataset.clear()

    print('Accuracy on test set: ', test_acc)

    np.savez(os.path.join(classifier_dir, 'val.npz'), test_acc=test_accs)
