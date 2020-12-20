import torch
import numpy as np
import os
import trimesh
from tqdm import tqdm
import config_incr as config
from dataloader_incr import Dataset
# from dataloader_genre import Dataset

from model import OccupancyNetwork
from torch.autograd import Variable
import torch.optim as optim
from classifier_model import SDFNetClassifier
import utils


def main():
    # torch.backends.cudnn.benchmark = True
    out_dir = config.training['out_dir']
    algo = config.training['algo']
    model_selection_path = config.testing['model_selection_path']
    cont = config.training['cont']
    # if model_selection_path is not None:
    #     model_selection_path = os.path.join(out_dir, model_selection_path)
    #     model_selection = np.load(model_selection_path, allow_pickle=True)
    #     ep = model_selection['epoch']
    #     model_path = 'model-%s.pth.tar'%(ep)
    #     model_path = os.path.join(out_dir, model_path)
    # else:
    #     if algo == 'occnet':
    #         if cont is None:
    #             model_path = os.path.join(out_dir, 'best_model.pth.tar')
    #         else:
    #             model_path = os.path.join(out_dir, 'best_model_cont.pth.tar')

    #     elif algo == 'disn':
    #         if cont is None:
    #             model_path = os.path.join(out_dir, 'best_model_iou.pth.tar')
    #         else:
    #             model_path = os.path.join(out_dir, 'best_model_iou_cont.pth.tar')


    # print('Loading model from %s'%(model_path))



    eval_task_name = config.testing['eval_task_name']
    eval_dir = os.path.join(out_dir, 'eval')
    eval_task_dir = os.path.join(eval_dir, eval_task_name)
    os.makedirs(eval_task_dir, exist_ok=True)
    mdl = config.training['model']

    batch_size_test = config.testing['batch_size_test']
    coord_system = config.training['coord_system']

    box_size = config.testing['box_size']

    split_counter = config.testing['split_counter']+1

    joint = config.training['joint']
    nclass = config.training['nclass']

    # Dataset
    print('Loading data...')
    test_dataset = Dataset(config, mode='test', algo=algo, coord_system=coord_system)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size_test, num_workers=12,pin_memory=True)
    all_classes_orig = test_dataset.catnames

    # import pdb; pdb.set_trace()
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
       # current_counter = prev_rep*len(all_classes_orig)
       try:
           current_counter = int(cont.split('-')[1])+1
       except Exception:
           print('Current counter is not an integer')


    # import pdb; pdb.set_trace()

    # Loading model
    if not joint:
        model = OccupancyNetwork(config, model=mdl)
    else:
        shape_model = OccupancyNetwork(config, model=mdl)
        model = SDFNetClassifier(model=shape_model)
    model = torch.nn.DataParallel(model).cuda()

    # if model_selection_path is not None:
    #     checkpoint = torch.load(model_path)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    # else:
    #     model.load_state_dict(torch.load(model_path))
    # model = torch.nn.DataParallel(model).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)


    out_obj_cat_all = []
    out_pose_all = []
    out_cd_all = []
    out_normals_all = []
    out_iou_all = []
    out_fscore_all = []
    out_acc_all = []

    # if not cont is None:
        # out_path = os.path.join(eval_task_dir, 'out-%s.npz'%(split_counter))
        # if os.path.exists(out_path):
        #     out_data = np.load(out_path, allow_pickle=True)
        #     out_obj_cat_all = list(out_data['obj_cat'])
        #     out_pose_all = list(out_data['pose'])
        #     out_cd_all = list(out_data['cd'])
        #     out_normals_all = list(out_data['normals'])
        #     out_iou_all = list(out_data['iou'])
        #     out_fscore_all = list(out_data['fscore'])

        #     seen_classes = list(out_data['seen_classes'])
        #     # seen_classes = list(set(all_classes[:current_counter]))
        # else:
        #     current_counter = 0
    seen_classes = list(set(all_classes[:current_counter].reshape(-1)))
    # import pdb; pdb.set_trace()
    print('Start counter: ', current_counter)
    print('End counter: ', split_counter-1)
    # import pdb; pdb.set_trace()
    for cl_count, cl_group in enumerate(all_classes[current_counter:split_counter]):
        print('Num seen classes: ', len(seen_classes))
        cl_count += current_counter
        # Load model and reload loader
        if algo == 'disn':
            model_path = 'best_model_iou_train-%s.pth.tar'%(cl_count)
        elif algo == 'occnet':
            model_path = 'best_model_train-%s.pth.tar'%(cl_count)

        new_classes = []
        for cl in cl_group:
            if cl not in seen_classes:
                seen_classes.append(cl)
                new_classes.append(cl)
        if joint:
            # import pdb; pdb.set_trace()

            model.module.increment_class(new_classes)
            model = torch.nn.DataParallel(model.module).cuda()
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
            print('Evaluating exposure %s, learned class %s, evaluating class %s'%(cl_count, cl, seen_classes[s]))

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

                        if joint:
                            _, cl_logits = model(points_input, img_input)

                            pred_labels = torch.argmax(cl_logits, dim=1)
                            cl_acc = torch.sum(pred_labels == labels).item()
                            out_acc.append(cl_acc)
                            # import pdb; pdb.set_trace()


                        obj, cat = obj_cat
                        cat_path = os.path.join(eval_task_dir, cat[0])

                        os.makedirs(cat_path, exist_ok=True)
                        if algo == 'occnet':
                            mesh = utils.generate_mesh(img_input, points_input, \
                                model.module)
                            obj_path = os.path.join(cat_path, '%s.obj' % obj[0])
                            mesh.export(obj_path)
                        elif algo == 'disn':
                            obj_path = os.path.join(cat_path, '%s-%s.obj' % (cl_count, obj[0]))
                            sdf_path = os.path.join(cat_path, '%s-%s.dist' % (cl_count, obj[0]))
                            if not joint:
                                mesh = utils.generate_mesh_mise_sdf(img_input, \
                                    points_input, model.module, box_size=box_size,\
                                    upsampling_steps=2, resolution=64)
                                # import pdb; pdb.set_trace()
                                mesh.export(obj_path)
                                # utils.generate_mesh_sdf(img_input, model.module, obj_path, sdf_path,\
                                    # box_size=box_size)
                            else:
                                utils.generate_mesh_sdf(img_input, model.module.rec_model, obj_path, sdf_path,\
                                    box_size=box_size)
                            # import pdb; pdb.set_trace()

                        # Save gen info
                        out_obj_cat.append(obj_cat)
                        out_pose.append(pose)

                        
                        # Calculate metrics
                        if algo == 'occnet':
                            out_dict = utils.eval_mesh(mesh, pointclouds, normals,\
                                        points_input, values)
                            # logits = model(points_input, img_input)
                            # model_iou = utils.compute_iou(logits.detach().cpu().numpy(), \
                            #             values.detach().cpu().numpy())
                            # print('Model iou: ', model_iou)
                        elif algo == 'disn':
                            # load the mesh
                            if os.path.exists(obj_path):
                                # clean_obj_path = os.path.join(cat_path,\
                                #     '%s_clean.obj' % obj[0])
                                # utils.clean_mesh(obj_path, clean_obj_path)
                                #### Load clean mesh
                                # mesh = trimesh.load(clean_obj_path)

                                #### Load unclean mesh
                                mesh = trimesh.load(obj_path)

                            else:
                                mesh = None
                            if not joint:
                                sdf_val = model(points_input, img_input)
                            else:
                                sdf_val,_ = model(points_input, img_input)

                            out_dict = utils.eval_mesh(mesh, pointclouds, normals, \
                                        points_input, values, algo='disn',\
                                        sdf_val=sdf_val)
                            # import pdb; pdb.set_trace()
                        
                        out_cd.append(out_dict['cd'])
                        out_normals.append(out_dict['normals'])
                        out_iou.append(out_dict['iou'])
                        out_fscore.append(out_dict['fscore'])
                        pbar.update(1)


            print('======> CD: ', np.mean(out_cd))
            if algo == 'disn':
                print('======> IOU: ', np.mean(out_iou,axis=0)[0])
            elif algo == 'occnet':
                print('======> IOU: ', np.mean(out_iou,axis=0))

            print('======> NORMALS: ', np.mean(out_normals))
            print('======> FSCORE: ', np.mean(out_fscore,axis=0)[1])
            if joint:
                print('======> ACC: ', np.mean(out_acc,axis=0))

            out_obj_cat_cl.append(out_obj_cat)
            out_pose_cl.append(out_pose)
            out_cd_cl.append(out_cd)
            out_normals_cl.append(out_normals)
            out_iou_cl.append(out_iou)
            out_fscore_cl.append(out_fscore)
            if joint:
                out_acc_cl.append(out_acc)

        out_obj_cat_all.append(out_obj_cat_cl)
        out_pose_all.append(out_pose_cl)
        out_cd_all.append(out_cd_cl)
        out_normals_all.append(out_normals_cl)
        out_iou_all.append(out_iou_cl)
        out_fscore_all.append(out_fscore_cl)
        if joint:
            out_acc_all.append(out_acc_cl)
        if not joint:
            np.savez(os.path.join(eval_task_dir, 'out-%s.npz'%(split_counter)), \
                obj_cat=np.array(out_obj_cat_all), pose=np.array(out_pose_all),\
                cd=np.array(out_cd_all), normals=np.array(out_normals_all),\
                iou=np.array(out_iou_all), fscore=np.array(out_fscore_all),\
                all_classes=all_classes, seen_classes=seen_classes)
        else:
            np.savez(os.path.join(eval_task_dir, 'out-%s.npz'%(split_counter)), \
                obj_cat=np.array(out_obj_cat_all), pose=np.array(out_pose_all),\
                cd=np.array(out_cd_all), normals=np.array(out_normals_all),\
                iou=np.array(out_iou_all), fscore=np.array(out_fscore_all),\
                acc=np.array(out_acc_all), all_classes=all_classes, seen_classes=seen_classes)
                    



if __name__ == '__main__':
    main()



