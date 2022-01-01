import torch.nn as nn
import torch.nn.functional as F

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 16, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        
        ## decoder layers ##
        self.t_conv1 = nn.ConvTranspose2d(16, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 16, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(16, 16, 2, stride=2)
        self.t_conv4 = nn.ConvTranspose2d(16, 3, 2, stride=2)

        self.classes_map = {}
        self.classes = []
        self.n_known = 0
        self.n_classes = 0

    def forward(self, x):
        ## encode ##
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = F.relu(self.conv4(x))
        x = self.pool(x)
        
        ## decode ##
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv3(x))
        x = F.relu(self.t_conv4(x))

                
        return x

    def increment_classes(self, new_classes):
        '''
        Add new output nodes when new classes are seen and make changes to
        model data members
        '''
        n = len(new_classes)
        
        self.n_classes += n

        for i, cl in enumerate(new_classes):
            self.classes_map[cl] = self.n_known + i
            self.classes.append(cl)
