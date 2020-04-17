import os
import cv2
import math
import tqdm
import torch
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
from torch.autograd import Variable
from tensorboardX import SummaryWriter
# from dlib_detector import DlibDetector
from torch.utils.tensorboard import SummaryWriter
# from utils import get_preds_fromhm, get_center_scale


class Trainer(object):
    def __init__(self, cuda, model, optimizer, criterion, train_loader, val_loader, out):
        self.cuda = cuda
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.out = out        
        self.writer = SummaryWriter("runs")
        self.epoch = 0
        self.max_epoch = 70
        self.iter_count=0
        self.val_iter_count=0
    
    def tensor_to_cv2(self,img):
        img = np.transpose(img, (1,2,0)).copy()#.astype(np.uint8).copy()#.astype(np.uint8)
        return img


    def draw_landmarks(self,img,landmarks_68,squeeze=False, clr=(255,0,0)):
        landmarks_68 = landmarks_68.cpu().numpy()
        # img = img.cpu().numpy()
        landmarks_68 = np.array(landmarks_68)
        if squeeze:
            landmarks_68 = landmarks_68.squeeze(0)
        for idx in range(len(landmarks_68)):
            y = landmarks_68[idx][0]
            x = landmarks_68[idx][1]
            # img= cv2.circle(img, (x, y), 2, clr, -1) 
            plt.scatter(x,y, s=5, color='blue')
        return img

    def draw_heatmap_landmarks(self,img,output,clr=(0,0,255)):
        for idx in range(68):
            hm = output[0,idx]
            (y,x) = np.unravel_index(hm.argmax(), hm.shape)
            y,x = int(x*4), int(y*4) 
            # img2 = cv2.circle(img, (x, y), 2, clr, -1) 
            plt.scatter(x,y, s=5, color='red')
        return img

    def get_landmarks_from_heatmaps(self,output):
        output_landmarks = []
        for item in output:
            landmarks = []
            for idx in range(68):
                hm = item[idx]
                (y,x) = np.unravel_index(hm.argmax(), hm.shape)
                x,y = int(x*4), int(y*4)
                landmarks.append([x, y])
            output_landmarks.append(landmarks)
        return torch.tensor(output_landmarks)

    # def draw_landmarks(self,img,landmarks_68,squeeze=False):
    #     landmarks_68 = landmarks_68.cpu()#.numpy()
    #     img = img.cpu().numpy()
    #     print(img.shape,landmarks_68.size())
    #     # if squeeze:
    #     #     landmarks_68 = landmarks_68.squeeze(0)
    #     for idx in range(68):
    #         # plt.scatter(y, x, s=6, color='blue')       
    #         # cv2.circle(imagetest, (x,y), 3,= 1 
    #         y = landmarks_68[idx][1]
    #         print(x,y)
    #         # imagetest=cv2.circle(imagetest, (y_hm,x_hm), 3, (255, 0, 0), -1)
    #         plt.scatter( y,x, s=6, color='red')
    #         # plt.scatter((y-9.04), (x+6.36), s=6, color='red')
    #     plt.imshow(img)
    #     plt.show()        
    #     return img

    def train_epoch(self):
        self.model.train()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        epochloss=0
        for batch_idx, (data) in tqdm.tqdm(enumerate(self.train_loader)): 
              
            input_data = data['image'].to(device)
            target = data['heatmap'].to(device)
            
            self.optimizer.zero_grad()
        
            output = self.model(input_data)[-1]             # Required shape  C*H*W 

            #- DEBUG CODE
            #-------------IN PLT--------------------
            # landmark_2d = data['landmarks_2d'].to(device)
            # input_data = self.draw_landmarks(input_data.cpu(),landmark_2d[0,0])
            # imgnew = self.draw_heatmap_landmarks(input_data,target.cpu())
            # imgnew = imgnew[0].squeeze(0).permute(1,2,0)
            # imgnew.cpu().numpy()
            # plt.imshow(imgnew)  
            # plt.show()
            #----------------------CV2-------------------  # Required shape H*W*C
            # landmark_2d = data['landmarks_2d'].to(device)
            # input_data = input_data.squeeze(0)
            # input_data = input_data.cpu().numpy()
            # imagenew = input_data.cpu()[0].squeeze(0).permute(0,1,2)#.numpy()  
            # image_cv2 = self.tensor_to_cv2(imagenew.numpy())                   
            # imagenew = self.draw_landmarks(image_cv2,landmark_2d[0,0]) 
            # # imagenew = self.draw_heatmap_landmarks(imagenew,target.cpu())
            # imagenew = self.draw_heatmap_landmarks(imagenew,output.cpu())            
            # dst = imagenew.copy()
            # cv2.transpose(imagenew,dst)
            # cv2.imshow("InputImage",dst)
            # cv2.waitKey(100)   
            #- DEBUG CODE                     
            
           
            loss = self.criterion(output, target).to(device)           
            clone_loss = loss.clone()
            loss = clone_loss/len(input_data)
            loss_data = loss.data.item()
            loss.backward()
            self.optimizer.step()
            
            # with open(osp.join(self.out, 'train_log.csv'), 'a') as f:
            #     log = "Epoch :"+ str(self.epoch)+ "Iteration :"+ str(batch_idx)+ "Loss : "+ str(loss_data)+ '\n'
            #     f.write(log)
            print('epoch: \t', self.epoch, '\t Training Loss: \t', loss_data)
            self.writer.add_scalar('Loss/train', loss_data, self.iter_count)
            self.iter_count=self.iter_count+1
            self.writer.flush()
            epochloss+=loss_data/4
        self.writer.add_scalar('Epoch_Loss/train',(epochloss/(batch_idx+1)), self.epoch)


        if self.epoch % 10 == 0:
            self.validate()

        # if self.epoch % 10 == 0:
        #     self.test()


    def validate(self):
        
        self.model.eval()
        # writer = SummaryWriter(log_dir='./vallogs/')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        epochloss=0
        for batch_idx, (data) in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='Validation epoch=%d' % self.epoch, ncols=80, leave=False):
        
            input_data = data['image']
            target = data['heatmap']
           
            # if self.cuda:
            #     data = data.cuda()
            input_data = input_data.to(device)
            
            target = target.to(device)

            with torch.no_grad():
                output = self.model(input_data)[-1]
            
            loss = self.criterion(output, target).to(device)
            loss_data = loss.data.item()
            
            print('epoch: \t', self.epoch, '\t Validation Loss: \t', loss_data)

            checkpoint = {'epoch': self.epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optim_state_dict' : self.optimizer.state_dict()}

            out_path = osp.join('./Results', 'Epoch'+str(self.epoch))
            if not osp.exists(out_path):
                os.mkdir(out_path)
            torch.save(checkpoint, out_path+'/checkpoint.pth.tar')

            self.writer.add_scalar('Loss/val', loss_data, self.val_iter_count)
            self.val_iter_count=self.val_iter_count+1

            epochloss+=loss_data/4 

        self.writer.add_scalar('Epoch_Loss/val', (epochloss/(batch_idx+1)), self.epoch)

        self.writer.flush()
        # writer.close()
                
    def test(self):
        self.model.eval()

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        self.model.to(device)
        total_error = 0
        for batch_idx, (data) in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='Test epoch=%d' % self.epoch, ncols=80, leave=False):
            
            input_data = data['image']
            target = data['heatmap']
            bounding_box = data['bounding_box']
            target_landmarks = data['landmarks_2d'].squeeze(1)
            
            output = self.model(input_data)[-1]

            output_landmarks = self.get_landmarks_from_heatmaps(output)

            w = abs(bounding_box[0] - bounding_box[2])*1.0
            h = abs(bounding_box[1] - bounding_box[3])*1.0
            d = torch.sqrt(w*h)

            total_error += self.mean_error(output_landmarks, target_landmarks, d)     
        NME = total_error/2954

        print("NME for epoch ", str(self.epoch), "is ", str(NME))
        self.writer.add_scalar('Accuracy/NME', NME, self.epoch)
        # self.writer.flush()
        # writer.close()

    def mean_error(self, output_landmarks, target_landmarks, d):
        mean_error = 0.0

        for idx, (output, target) in enumerate(list(zip(output_landmarks, target_landmarks))):

            output = output.permute(1, 0) #[x1, y1]
            target = target.permute(1, 0) #[x2, y2]

            diff_x = abs(output[0] - target[0])
            diff_y = abs(output[1] - target[1])

            distances = torch.sqrt(diff_x**2 + diff_y**2)
            sum_distances = torch.sum(distances)
            mean_error += (sum_distances/d[idx])
            
        return mean_error

    def train(self):

        for epoch in tqdm.trange(self.epoch, self.max_epoch, desc='Train', ncols=80):
            self.epoch = epoch
            self.train_epoch()

        self.writer.close()
    
