import os
import cv2
import sys
import json
import dlib
import math
import torch
import glob
import itertools
import numpy as np
from PIL import Image
import torch.nn as nn
from Track import Test_FAN
import torch.optim as optim
import matplotlib.pyplot as plt
from models import FAN, ResNetDepth
from dlib_detector import DlibDetector
from torchvision import transforms, utils
from utils import crop_sample,create_shriya_heatmap_version
from torch.utils.data import Dataset, DataLoader, random_split


class Menpo(Dataset):

    def __init__(self, root_path, transform=transforms.Compose([transforms.ToTensor()])):  #read image path and model space, projected space

        self.images = []
        self.model_space = []
        self.projected_space = []
        self.root_path = root_path
        

        for entry in sorted(glob.glob('/home/kak/Documents/DFKI/Tracking/0038/images/*.jpg')):
        # for entry in os.scandir(root_path):
            # if entry.path.endswith('.jpg'):
            self.images.append(entry.split('/')[-1])
        
        # model_space_path = '/home/kak/Documents/DFKI/'
        projected_space_path = '/home/kak/Documents/DFKI/Tracking/0038/projected_space/0038/'
        for i in range(len(self.images)):
            file_name = self.images[i].split('.')[0]
            #self.model_space.append(model_space_path+file_name+'.ljson')
            self.projected_space.append(projected_space_path+file_name+'.ljson')
            self.transform=transform  
        self.face_detector = DlibDetector(device = 'cpu') 
        self.data_len = len(self.images)

    def __len__(self):
        return(len(self.images))

    def draw_landmarks(self,img,landmarks_68,squeeze=False, clr=(255,0,0)):
        landmarks_68 = np.array(landmarks_68)
        if squeeze:
            landmarks_68 = landmarks_68.squeeze(0)
        for idx in range(len(landmarks_68)):
            x = landmarks_68[idx][0]
            y = landmarks_68[idx][1]
            img = cv2.circle(img, (x, y), 2, clr, -1) 
        return img

    def draw_heatmap_landmarks(self,img,output,clr=(0,0,255)):
        for idx in range(68):
            hm = output[0,idx]
            (x,y) = np.unravel_index(hm.argmax(), hm.shape)
            x,y = int(x*4), int(y*4) 
            img = cv2.circle(img, (x, y), 2, clr, -1) 
        return img

    def get_landmarkface(self, detected_faces,landmarks):
        landmarks = np.array(landmarks)
        means = np.sum(landmarks, axis=0)/68
        x_mean = means[0]
        y_mean = means[1]
        face_index = 0
        compare_distance = sys.maxsize
        for i, d in enumerate(detected_faces):
            center = torch.FloatTensor([d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
            center[1] = center[1] - (d[3] - d[1]) * 0.12
            distance = math.sqrt((center[0] - x_mean)**2 + (center[1]- y_mean)**2)
            if distance < compare_distance:
                face_index = i
                compare_distance = distance
        return face_index

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_name = self.images[idx]
        img_path = self.root_path + image_name
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        
        image_size_original = image.shape 

        #---------------DLIB DETECTOR-----------------------------------------------
        detected_faces = self.face_detector.detect_from_image(image[..., ::-1].copy())
        if len(detected_faces) < 1:
            return self.__getitem__(np.random.randint(low=0, high=self.data_len))             
        
        
        #model_space_json = self.model_space[idx]
        projected_space_json = self.projected_space[idx]
        '''
        Read the information from json files
        '''
        #with open(model_space_json) as json_file:
         #   model_space_data = json.load(json_file)['landmarks']
        
        with open(projected_space_json) as json_file:
            projected_space_data = json.load(json_file)['landmarks']
         
        landmarks_68 = []
        #model_space_landmarks_3d = model_space_data['points']  #[:68] # information from model_space_json file
        projected_space_landmarks_2d = projected_space_data['points'] # information from projected_space_json file
        
        for i in range(len(projected_space_landmarks_2d)):
            if(i<=32):
                if(i%2 == 0):
                    landmarks_68.append(projected_space_landmarks_2d[i])
            else:
                landmarks_68.append(projected_space_landmarks_2d[i])
           
        
        face_index = 0        
        if len(detected_faces) >= 1 :
            face_index = self.get_landmarkface( detected_faces,landmarks_68)

        
        landmarks_68 = torch.tensor([[x, y] for i,(x, y) in enumerate(landmarks_68)])
        # plotting ground truth from dataset
        # img = self.draw_landmarks(image,landmarks_68)

        landmarks_68 = torch.FloatTensor(landmarks_68).unsqueeze(0)
        image = np.array(image)

        #-----------------------Detector--------------------------        
        d = detected_faces[face_index]
        center2 = torch.FloatTensor([d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
        center2[1] = center2[1] - (d[3] - d[1]) * 0.12
        scale2 = ((d[2] - d[0]) + (d[3] - d[1])) / self.face_detector.reference_scale
        
        image2,landmarks_68 = crop_sample(image,landmarks_68,center2,scale2)           

        img_size = image2.shape[0]
        image3 = cv2.resize(image2, dsize=(int(256), int(256)),interpolation=cv2.INTER_LINEAR)
        scale = 256/img_size 
        landmarks_68 = np.array(landmarks_68) * scale 
        heatmaps = create_shriya_heatmap_version(landmarks_68/4)
        heatmaps = heatmaps.squeeze(0).type(torch.FloatTensor)
        
        image3 = np.float32(image3)
        image3 = image3/255   # Normalizing
        
        image3 = self.transform(image3) #.permute(1,2,0) only for plotting 
        
        # image3 = torch.FloatTensor(image3)
        
        # plt.imshow(image3)
        # plt.show()
        ''' 
        todo
        Choose 68 points from the above 84 points, evenly.
        '''
        item = {
            'image':image3,
            'image_name':image_name,
            #'model_space_landmarks':model_space_landmarks_3d,
            'projected_space_landmarks':projected_space_landmarks_2d,
            'landmarks_2d':landmarks_68,
            'heatmap':heatmaps,
            'bounding_box':d,
            'scale': scale2
        }
        
        return item

test_data = Menpo(root_path='/home/kak/Documents/DFKI/Tracking/0038/images/')
# test_data = Menpo(root_path='/home/kak/Documents/DFKI/MenpoTracking/Menpo_Challenge/test/')

# val_data, train_data = random_split(train_data,[2954,6000])

# train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=0)

test_loader = DataLoader(test_data, batch_size=1, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1 = FAN(4)
# fan_weights1 = torch.load('checkpoint_60_dlib.pth.tar', map_location='cuda')
# model1.load_state_dict(fan_weights1['model_state_dict'])
fan_weights1 = torch.load('/home/kak/Documents/DFKI/Resnet/PretrainedModels/3DFAN4-7835d9f11d.pth.tar', map_location='cuda')
model1.load_state_dict(fan_weights1)


model2 = ResNetDepth()
# depth_weights = torch.load('/home/kak/Documents/DFKI/TestingFAN/checkpoint_resnet_usingbb_3d.pth.tar', map_location='cuda')
depth_weights = torch.load('/home/kak/Documents/DFKI/FaceAlignmentNet/PretrainedModels/depth-2a464da4ea.pth.tar', map_location='cuda')

depth_dict = { k.replace('module.', ''): v for k,
                v in depth_weights['state_dict'].items()}
model2.load_state_dict(depth_dict)

# criterion = nn.MSELoss(reduction='mean')
# optimizer = optim.Adam(network_FAN.parameters(), lr=1e-6)


cuda = torch.cuda.is_available()
out = './Result/'

test = Test_FAN(cuda, model1, model2, test_loader, out)
test.test()
