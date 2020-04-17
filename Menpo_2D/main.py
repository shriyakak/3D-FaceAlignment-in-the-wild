import torch 
import torch.nn as nn
import torch.optim as optim
from Dataloader_bb import Menpo
from train_menpo import Trainer
from models import FAN, ResNetDepth
from torch.utils.data import Dataset, DataLoader, random_split

train_data = Menpo(root_path='./Menpo_Challenge/Train_data/')

val_data = Menpo(root_path ='./Menpo_Challenge/Val_data/')

# val_data, train_data = random_split(train_data,[2954,6000])


train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=2)

val_loader = DataLoader(val_data, batch_size=8, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
network_FAN = FAN(4)
fan_weights = torch.load('./FAN_model.pth.tar', map_location='cuda')
network_FAN.load_state_dict(fan_weights['model_state_dict'])

criterion = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(network_FAN.parameters(), lr=1e-5)

cuda = torch.cuda.is_available()
out = './Result/'

trainer = Trainer(cuda, network_FAN, optimizer, criterion, train_loader, val_loader, out)
trainer.train()
