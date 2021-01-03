from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import numpy as np
import torch
from PIL import Image
import cv2
import time
import copy
from collections import Counter

class iCIFAR10(CIFAR10):
    def __init__(self, args, root, classes,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False,
                 mean_image=None):
        """
        Args : 
            args : arguments from the argument parser
            classes : list of groundtruth classes
            train : Whether the model is training or testing
            transform : Image transformation performed on train set
            target_transform : Image transformation performed on test set
            download : Whether to download from the source
            mean_image : the mean image over the train dataset           
        """
        # Inherits from CIFAR10, where self.train_data and self.test_data are of type uint8, dimension 32x32x3
        super(iCIFAR10, self).__init__(root,
                                       train=train,
                                       transform=transform,
                                       target_transform=target_transform,
                                       download=download)

        self.img_size = args.img_size
        self.aug = args.aug
        # Number of frame at each learning exposure
        self.num_e_frames = args.lexp_len
        # Whether to sample data with replacement
        self.s_w_replace = args.sample_w_replacement
        self.num_exemplars = args.num_exemplars
        # weights for minibatch sampler
        self.sample = args.sample

        # Select a subset of classes for incremental training and testing
        if self.train:
            self.train_data = self.data
            self.train_labels = self.targets
            # Resize and transpose to CxWxH all train images
            resized_train_images = np.zeros((len(self.train_data), 3, \
                self.img_size, self.img_size), dtype=np.uint8)
            for i, train_image in enumerate(self.train_data):
                resized_train_images[i] = cv2.resize(train_image, \
                    (self.img_size, self.img_size)).transpose(2,0,1)
            self.train_data = resized_train_images
            # if mean_image is None, compute mean image from the train set and save the mean image
            if mean_image is None:
                self.mean_image = np.mean(np.float32(self.train_data), axis=0)
                np.save("data_generator/cifar_mean_image.npy", self.mean_image)
            else:
                self.mean_image = mean_image
            self.all_train_data = self.train_data
            self.all_train_labels = self.train_labels
            self.all_train_coverage = np.zeros(len(self.all_train_labels), \
                                               dtype=np.bool_)
            if self.aug == "icarl" or self.aug == "e2e":
                self.train_data, self.train_labels = [], [] 
                # e_maps keeps track of new images in the current learning exposure with regard to images from the exemplar set
                self.e_maps = -np.ones((self.num_e_frames,2), dtype=np.int32)
                self.weights = np.ones(self.num_e_frames, dtype=float)

            elif self.aug == "e2e_full":
                self.train_data, self.train_labels = np.zeros((12*(self.num_e_frames+self.num_exemplars),3,self.img_size,self.img_size), dtype=np.uint8), []
                self.curr_len = 0
                self.curr_orig_len = 0
                self.e_maps = -np.ones((12*(self.num_e_frames+self.num_exemplars),2), dtype=np.int32)
                self.weights = np.ones(12*(self.num_e_frames+self.num_exemplars), dtype=float)



        else:
            self.test_data = self.data
            self.test_labels = self.targets
            # Resize all test images
            resized_test_images = np.zeros((len(self.test_data), 3, \
                self.img_size, self.img_size), dtype=np.uint8)
            for i, test_image in enumerate(self.test_data):
                resized_test_images[i] = cv2.resize(test_image, \
                    (self.img_size, self.img_size)).transpose(2,0,1)
            self.test_data = resized_test_images
            # Load mean image if mean_image is not parsed in
            if mean_image is None:
                self.mean_image = np.load( \
                    "data_generator/cifar_mean_image.npy")
            else:
                self.mean_image = mean_image
            self.all_test_data = self.test_data
            self.all_test_labels = self.test_labels
            self.test_data, self.test_labels = [], []

    def __getitem__(self, index):
        # Data and mean image of dimension 3 x img_size x img_size, unormalized
        if self.train:
            if self.aug == 'icarl':
                img = self.train_data[index].transpose(1,2,0)
                img = Image.fromarray(np.uint8(img))
                img = np.array(self.transform(img))
                img = img.transpose(2,0,1)

                img = (img - self.mean_image)/255.
                # Augment : Random crops and horizontal flips
                random_cropped = np.zeros(img.shape, dtype=np.float32)
                padded = np.pad(img, ((0, 0), (4, 4), (4, 4)), 
                                mode="constant")
                crops = np.random.random_integers(0, high=8, size=(1, 2))
                if (np.random.randint(2) > 0):
                    random_cropped[:, :, :] = padded[:, 
                        crops[0, 0]:(crops[0, 0] + self.img_size), 
                        crops[0, 1]:(crops[0, 1] + self.img_size)]
                else:
                    random_cropped[:, :, :] = padded[:, 
                        crops[0, 0]:(crops[0, 0] + self.img_size), 
                        crops[0, 1]:(crops[0, 1] + self.img_size)][:, :, ::-1]

                img = torch.FloatTensor(random_cropped)         

                target = self.train_labels[index]
            elif self.aug == "e2e":
                # Augment : 
                # For 3 choices - original image, brightness augmented 
                # and contrast augmented
                curr_img = np.float32(self.train_data[index])
                curr_mean = self.mean_image
                rand_num = np.random.randint(0, 3)
                if rand_num == 1:
                    # brightness
                    brightness = np.random.randint(-63, 64)
                    curr_img = np.clip(curr_img + brightness, 0, 255)
                elif rand_num == 2:
                    # contrast
                    # random from 0.2 to 1.8 inclusive
                    contrast = np.random.random() * (1.6+1e-16) + 0.2  
                    curr_img = np.clip((curr_img-curr_mean) * contrast 
                                        + curr_mean, 0, 255)

                curr_img = (curr_img - curr_mean)/255.
                    
                # Random choice whether to crop
                rand_num = np.random.randint(0, 2)
                if rand_num == 1:
                    crops = np.random.random_integers(0, high=8, size=2)
                    padded = np.pad(curr_img,((0,0),(4,4),(4,4)), 
                                    mode='constant')
                    curr_img = padded[:,
                                      crops[0]:(crops[0]+self.img_size),
                                      crops[1]:(crops[1]+self.img_size)] 

                # Random choice whether to mirror horizontally
                rand_num = np.random.randint(0, 2)
                if rand_num == 1:
                    curr_img = copy.deepcopy(curr_img[:, :, ::-1])
                img = torch.FloatTensor(curr_img)
                target = self.train_labels[index]
            elif self.aug == "e2e_full":
                # Data augmentation as in E2EIL original algorithm
                img = self.train_data[index]
                img = (img - self.mean_image)/255.
                target = self.train_labels[index]
            weight = self.weights[index]

        else:
            img = self.test_data[index]
            img = (img - self.mean_image)/255.
            
            target = self.test_labels[index]

        img = torch.FloatTensor(img)
        target = np.array(target)

        if self.train:
            return index, img, target, weight
        
        return index, img, target

    def __len__(self):
        if self.train:
            if self.aug == "icarl" or self.aug == "e2e":
                return len(self.train_data)
            elif self.aug == "e2e_full":
                return self.curr_len
        else:
            return len(self.test_data)

    def load_data_class(self, classes, model_classes, iteration):
        """Loads train data, label and e_maps for current learning exposure
        Args : 
            classes : List of groudtruth classes
            model_classes : List of the classes that the model sees
            iteration : Learning exposure the model is on          
        """
        # called in train only
        if self.train:
            train_data = []
            train_labels = []
            if self.aug == "icarl" or self.aug == "e2e":
                self.e_maps = -np.ones((self.num_e_frames,2), dtype=np.int32)
            elif self.aug == "e2e_full":
                self.train_data, self.train_labels = np.zeros((12*(self.num_e_frames+self.num_exemplars),3,self.img_size,self.img_size), dtype=np.uint8), []
                self.curr_len = 0
                self.e_maps = -np.ones((12*(self.num_e_frames+self.num_exemplars),2), dtype=np.int32) 
            for (gt_label, model_label) in zip(classes, model_classes):
                rand = np.random.choice(500, self.num_e_frames, replace = self.s_w_replace)

                s_ind = np.where( \
                    np.array(self.all_train_labels) == gt_label)[0]

                s_images = self.all_train_data[s_ind[rand]]

                s_labels = np.array(self.all_train_labels)[s_ind[rand]]

                train_data.append(s_images)
                train_labels.append(np.array([model_label]*len(s_images)))

                self.all_train_coverage[s_ind[rand]] = True

                self.e_maps[:self.num_e_frames,0] = iteration
                self.e_maps[:self.num_e_frames,1] = s_ind[rand]

            if self.aug == "icarl" or self.aug == "e2e":
                self.train_data = np.concatenate(np.array(train_data, \
                    dtype=np.uint8),axis=0)
                self.train_labels = np.concatenate(np.array(train_labels, \
                    dtype=np.int32), axis=0).tolist()
            elif self.aug == 'e2e_full':
                self.train_data[:self.num_e_frames] = np.concatenate(np.array(train_data, \
                    dtype=np.uint8),axis=0)
                self.train_labels = np.concatenate(np.array(train_labels, \
                    dtype=np.int32), axis=0).tolist()
                self.curr_len = self.num_e_frames
                self.curr_orig_len = self.num_e_frames 


    def expand(self, model_new_classes, gt_new_classes):
        """Expands current test set if new classes are seen
        Args : 
            model_new_classes : List of new classes that the model sees
            gt_new_classes : List of the classes that the model sees        
        """
        # calls in test only

        if not self.train:
            test_data = []
            test_labels = []

            for (mdl_label, gt_label) in \
                    zip(model_new_classes, gt_new_classes):
                s_images = self.all_test_data[\
                            np.array(self.all_test_labels) == gt_label]
                test_data.append(s_images)
                test_labels.append(np.array([mdl_label]*len(s_images)))

                    
            if len(test_data) > 0:
                test_data = np.concatenate( \
                    np.array(test_data, dtype=np.uint8),axis=0)
                test_labels = np.concatenate( \
                    np.array(test_labels, dtype=np.uint8), axis=0)
            if len(self.test_data) == 0:
                self.test_data = test_data
                self.test_labels = test_labels.tolist()
            else:
                if len(test_data) > 0:
                    self.test_data = np.concatenate( \
                        [self.test_data, test_data], axis=0)
                    self.test_labels = np.concatenate( \
                        [self.test_labels, test_labels], axis=0).tolist()

    def get_train_coverage(self, label):
        """Returns the coverage of requested label. Coverage is calculated based on the number of images the model has seen for this label / the total number of images of this label
        Args:
            label : The requested label
        """
        num_images_label = len(self.all_train_coverage[np.array(self.all_train_labels) == label])
        num_images_covered = self.all_train_coverage[np.array(self.all_train_labels) == label].sum()

        return num_images_covered*100./ num_images_label

    def get_image_class(self, label):
        """Returns the images and e_maps of the requested label
        Args:
            label : The requested label
        """
        if self.aug == "e2e_full":
            return self.train_data[:self.curr_orig_len][ \
                        np.array(self.train_labels)[:self.curr_orig_len] == label], \
                        self.e_maps[:self.curr_orig_len][np.array(self.train_labels)[:self.curr_orig_len] == label]
        return self.train_data[ \
                    np.array(self.train_labels) == label], \
                    self.e_maps[np.array(self.train_labels) == label]

    def append(self, images, labels, e_map_data):
        """Appends dataset with images, labels and frame data from exemplars

        Args:
            images: Tensor of shape (N, C, H, W)
            labels: list of labels
            e_map_data: frame data of exemplars
        """
        if self.aug == "icarl" or self.aug == "e2e":
            self.train_data = np.concatenate((self.train_data, images), axis=0)
            self.train_labels = self.train_labels + labels
            self.e_maps = np.concatenate((self.e_maps, e_map_data), axis=0)
            self.weights = np.concatenate((self.weights, np.ones(len(labels), dtype=float)))
        elif self.aug == "e2e_full":
            self.train_data[self.curr_orig_len:self.curr_orig_len+len(labels)] = images
            self.train_labels = self.train_labels + labels
            self.e_maps[self.curr_orig_len:self.curr_orig_len+len(labels)] = e_map_data
            self.curr_orig_len += len(labels)
            self.curr_len += len(labels)

    def update_class_weights(self):
        if self.aug == "icarl" or self.aug == "e2e":
            self.weights = np.ones(self.weights.shape, dtype=float)
            cnt = Counter(self.train_labels)
            curr_labels = self.train_labels
            least_common_class, cnt_least_common = cnt.most_common(len(cnt))[-1]

            for i, label in enumerate(curr_labels):
                cnt_lbl = cnt[label]
                self.weights[i] = float(cnt_least_common)/cnt_lbl
        elif self.aug == "e2e_full":
            self.weights = np.ones(self.weights.shape, dtype=float)
            cnt = Counter(self.train_labels[:self.curr_len])
            curr_labels = self.train_labels[:self.curr_len]
            least_common_class, cnt_least_common = cnt.most_common(len(cnt))[-1]
            for i, label in enumerate(curr_labels):
                cnt_lbl = cnt[label]
                self.weights[:self.curr_len][i] = float(cnt_least_common)/cnt_lbl

    def get_augmented_set(self):
        """
        Gets augmented training set, each image has 12 copies, 11 modified and itself
        """
        train_data = []
        train_labels = []

        for i in range(self.curr_orig_len):
            start_len = self.curr_len
            curr_img = self.train_data[i]
            ################### data augmentation ################
            # brightness
            brightness = np.random.randint(-63, 64)
            b_curr_img = np.clip(np.int32(curr_img) + brightness, 0, 255)
            self.train_data[self.curr_len] = np.uint8(b_curr_img)
            self.curr_len+=1

            # contrast
            contrast = np.random.random() * (1.6+1e-16) + 0.2  # random from 0.2 to 1.8 inclusive
            c_curr_img = np.clip((np.float32(curr_img) - self.mean_image) * contrast + self.mean_image, 0, 255)
            self.train_data[self.curr_len] = np.uint8(c_curr_img)
            self.curr_len+=1

            # cropping
            crops = np.random.random_integers(0,high=8,size=(3,2))
            padded = list(np.pad(self.train_data[start_len:self.curr_len],((0,0),(0,0),(4,4),(4,4)),mode="constant"))
            padded.append(np.pad(curr_img, ((0,0),(4,4),(4,4)), mode="constant"))

            padded = np.array(padded)
            for r in range(len(padded)):
                self.train_data[self.curr_len:self.curr_len+1] = padded[r,:,crops[r,0]:(crops[r,0]+self.img_size),crops[r,1]:(crops[r,1]+self.img_size)]
                self.curr_len+=1

            # mirror
            num_images_transformed = self.curr_len - start_len
            self.train_data[self.curr_len:self.curr_len+num_images_transformed] = self.train_data[start_len:self.curr_len][:,:,:,::-1]
            self.curr_len += num_images_transformed
            self.train_data[self.curr_len:self.curr_len+1] = curr_img[:,:,::-1]
            self.curr_len += 1
            # normalize
            for img in range(11):
                ######################################################

                self.train_labels.append(self.train_labels[i])
            self.e_maps[start_len:self.curr_len,0] = np.array([self.e_maps[i,0]]*11)
            self.e_maps[start_len:self.curr_len,1] = np.array([self.e_maps[i,1]]*11)

class iCIFAR100(iCIFAR10):
    base_folder = "cifar-100-python"
    url = "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]
    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }