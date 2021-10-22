from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import numpy as np
import torch
from PIL import Image
import cv2
import time

class iCIFAR10(CIFAR10):
    def __init__(self, args, root, classes,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False,
                 mean_image=None,clb=False):
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
        # Inherits from CIFAR10, where self.train_data and self.test_data 
        # are of type uint8, dimension 32x32x3
        super(iCIFAR10, self).__init__(root,
                                       train=train,
                                       transform=transform,
                                       target_transform=target_transform,
                                       download=download)
        self.clb = clb
        self.img_size = args.img_size
        # Number of frame at each learning exposure
        self.num_e_frames = args.lexp_len

        self.num_classes = args.num_classes
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
            self.all_train_data = self.train_data
            print(self.all_train_data.shape)
            self.all_train_labels = self.train_labels
            self.all_train_coverage = np.zeros(len(self.all_train_labels), \
                                               dtype=np.bool_)
            self.train_data, self.train_labels, self.train_coverage = \
                                                [], [], []
            # e_maps keeps track of new images in the current learning 
            # exposure with regard to images from the exemplar set
            self.e_maps = -np.ones((self.num_e_frames*self.num_classes,2), dtype=np.int32)
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
            self.all_test_data = self.test_data
            print(self.all_test_data.shape)
            self.all_test_labels = self.test_labels
            self.test_data, self.test_labels = [], []

    def __getitem__(self, index):
        # Data and mean image of dimension 3 x img_size x img_size, unormalized
        if self.train:
            img = self.train_data[index]

            img = img/255.
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

            img = torch.FloatTensor(img)            

            target = self.train_labels[index]
        else:
            img = self.test_data[index]
            img = img/255.
            
            target = self.test_labels[index]

        img = torch.FloatTensor(img)
        target = np.array(target)
        
        return index, img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
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
            self.e_maps = -np.ones((self.num_e_frames*self.num_classes,2), dtype=np.int32)
            for i,(gt_label, model_label) in enumerate(zip(classes, model_classes)):
                rand = np.random.choice(500, self.num_e_frames, replace = False)

                s_ind = np.where( \
                    np.array(self.all_train_labels) == gt_label)[0]

                s_images = self.all_train_data[s_ind[rand]]

                s_labels = np.array(self.all_train_labels)[s_ind[rand]]

                train_data.append(s_images)
                train_labels.append(np.array([model_label]*len(s_images)))

                self.all_train_coverage[s_ind[rand]] = True

                self.e_maps[i*len(s_images):(i+1)*len(s_images),0] = iteration
                self.e_maps[i*len(s_images):(i+1)*len(s_images),1] = s_ind[rand]
            self.train_data = np.concatenate(np.array(train_data, \
                dtype=np.uint8),axis=0)
            self.train_labels = np.concatenate(np.array(train_labels, \
                dtype=np.int32), axis=0).tolist()

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
        """
        Returns the coverage of requested label. Coverage is calculated 
        based on the number of images the model has seen for this 
        label / the total number of images of this label
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
        self.train_data = np.concatenate((self.train_data, images), axis=0)
        self.train_labels = self.train_labels + labels
        self.e_maps = np.concatenate((self.e_maps, e_map_data), axis=0)

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

