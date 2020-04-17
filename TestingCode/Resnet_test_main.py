import torch 
import torch.nn as nn
import torch.optim as optim
from Resnet_dataloader import Menpo
# from menpo_train_resnet import Trainer
from Resnet_eval import Test_FAN
from models import FAN, ResNetDepth
from torch.utils.data import Dataset, DataLoader

#------------------------------- Dataset and DataLoaders -------------------------------------

# train_data = Menpo(root_path='/data/skak/project/MenpoTracking/Menpo_Challenge/Train_data/')
test_data = Menpo(root_path='/home/kak/Documents/DFKI/MenpoTracking/Menpo_Challenge/Test_data/')

# val_data = Menpo(root_path ='/data/skak/project/MenpoTracking/Menpo_Challenge/Val_data/')

# val_data, train_data = random_split(train_data,[2954,6000])


test_loader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=0)

# val_loader = DataLoader(val_data, batch_size=8, num_workers=0)

# #----------------------------------------------------------------------------------------------

# #------------------------- Creating Model and loading pretraind weights -----------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
network_FAN = FAN(4)
network_FAN = network_FAN.to(device)

network_depth = ResNetDepth()

fan_weights = torch.load('checkpoint_60_dlib.pth.tar', map_location='cuda')
network_FAN.load_state_dict(fan_weights['model_state_dict'])
depth_weights = torch.load('/home/kak/Documents/DFKI/TestingFAN/checkpoint_resnet_usingbb_3d.pth.tar', map_location='cuda')
#depth_weights = torch.load('/data/skak/project/FaceAlignmentNet/PretrainedModels/depth-2a464da4ea.pth.tar', map_location='cuda')

depth_dict = { k.replace('module.', ''): v for k,
                v in depth_weights['model_state_dict'].items()}
network_depth.load_state_dict(depth_dict)

# for params in network_depth.parameters():
#     params.requires_grad = True

network_FAN.cuda()
network_depth.cuda()


# # #----------------------------------------------------------------------------------------------

# # #--------------------------Criterion and Optimizer --------------------------------------------

# criterion = nn.MSELoss(reduction='mean').to(device)
criterion = nn.L1Loss().to(device)
# optimizer = optim.SGD(network_FAN.parameters(), lr=0.0001, momentum=0.9)
optimizer = optim.Adam(network_depth.parameters(), lr=0.0001)

# # #----------------------------------------------------------------------------------------------

cuda = torch.cuda.is_available()

out = './Result/'

test = Test_FAN(cuda, network_FAN, network_depth, test_loader, out)
test.test()
#---------------------------------------------------------------------------------------------
