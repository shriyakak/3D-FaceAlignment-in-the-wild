import torch
import cv2
import tqdm
import numpy as np
import collections
from models import FAN,ResNetDepth
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from utils import draw_gaussian,_gaussian, get_preds_fromhm, get_center_scale, create_shriya_heatmap_version


class Test_FAN(object):
    def __init__(self, cuda, model1, model2, test_loader, out):
        self.cuda = cuda
        self.model1 = model1
        self.model2 = model2
        # self.train_loader = train_loader
        # self.val_loader = val_loader
        self.test_loader = test_loader
        self.out = out        
        self.writer = SummaryWriter("runs")
        self.epoch = 0
        self.max_epoch = 1
        self.iter_count=0
        self.val_iter_count=0
    
    def get_landmarks_from_heatmaps(self,output):
        output_landmarks = []
        for item in output:
            landmarks = []
            for idx in range(68):
                hm = item[idx]
                (x,y) = np.unravel_index(hm.argmax(), hm.shape)
                x,y = int(x*4), int(y*4)
                landmarks.append([x, y])
            output_landmarks.append(landmarks)
        return torch.tensor(output_landmarks)
    def eval(self):
        self.model1.eval()
        self.model2.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")
        self.model1.to(device)
        self.model2.to(device)
        error_list1 = []
        # f1 = open('resnet.txt', 'w')
        # f2 = open('predicted_landmaks.txt', 'w')
        # total_error = 0

        for batch_idx, (data) in tqdm.tqdm(
                enumerate(self.test_loader), total=len(self.test_loader),
                desc='Test epoch=%d' % self.epoch, ncols=80, leave=False):


            # batch_size =  data.shape[0]
            # data = data.to(device)
            # scale = scale.to(device)

            input_data = data['image'].to(device)
            #z_coordinate = data['z_coordinate'].to(device)
            scale = data['scale']
            #landmarks_3D = data['landmarks_3D']
            bounding_box = data['bounding_box']
            image_name = data['image_name']
            # bounding_box = bounding_box.type(torch.FloatTensor)
            with torch.no_grad():

                out1 = self.model1(input_data)[-1]   

                out1 = out1.cpu()

                output_landmarks = self.get_landmarks_from_heatmaps(out1)
        
                output_landmarks = output_landmarks.to(device)

                heatmaps = create_shriya_heatmap_version(output_landmarks)

                image_concatenated_with_heatmap = torch.cat((input_data,heatmaps.to(device)),1)

                out2 = self.model2(image_concatenated_with_heatmap).to(device)
                # target_landmarks = z_coordinate.squeeze(1).to(device)

                output_landmarks = output_landmarks.to(torch.float64)
                print(output_landmarks.dtype)
                out2 = out2.cpu()
                out2 = out2 * (1.0 / (256.0 / (200.0 * scale)))
                out2 = out2.unsqueeze(2)
                out2 = out2.to(device)

            output_landmarks_orig = torch.cat((output_landmarks, out2), 2)       #[2, 68, 3]
            
            # bounding_box = bounding_box.squeeze(1)

            # bounding_box = bounding_box.permute(1, 0)

            w = abs(bounding_box[0] - bounding_box[2])*1.0
            h = abs(bounding_box[1] - bounding_box[3])*1.0
            d = torch.sqrt(w*h)

            output_landmarks = output_landmarks.cpu().squeeze(0).numpy()

            input_data = input_data.squeeze(0).permute(1,2,0).cpu()
            input_data = np.array(input_data)
            output_landmarks_orig = output_landmarks_orig.cpu()
            output_landmarks_orig = np.array(output_landmarks_orig).squeeze(0)
            plot_style = dict(marker='o',markersize=2, linestyle='-', lw=2)
            pred_type = collections.namedtuple('prediction_type', ['slice', 'color'])

            pred_types = {'face': pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
              'eyebrow1': pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
              'eyebrow2': pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
              'nose': pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
              'nostril': pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
              'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
              'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
              'lips': pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
              'teeth': pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))
              }
            # fig, ax = plt.figure()
            # ax =fig.add_subplot(1,2,)
            for d in range(68):
                x_2d = output_landmarks[d][0]
                y_2d = output_landmarks[d][1]
                plt.scatter(y_2d, x_2d, s=10, c='blue', alpha=0.5)
            plt.imshow(cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB))
            for pred_type in pred_types.values():
                plt.plot(output_landmarks_orig[pred_type.slice, 1], output_landmarks_orig[pred_type.slice, 0], color='white', **plot_style)
            plt.savefig('./mepotrackresult/0038/'+ image_name[0].split('.jpg')[0]+'.png')            
            # ax.axis('off')
            # plt.show()
            plt.clf()
            # ax = fig.add_subplot(1,2,2,projection='3d')
            
            # surf = ax.scatter(output_landmarks_orig[:, 0] * 1.2,
            #       output_landmarks_orig[:, 1],
            #       output_landmarks_orig[:, 2],
            #       c='green',
            #       alpha=1.0,
            #       edgecolor='b')
            # for pred_type in pred_types.values():
            #     ax.plot3D(output_landmarks_orig[pred_type.slice, 0] * 1.2, 
            #     output_landmarks_orig[pred_type.slice, 1], 
            #     output_landmarks_orig[pred_type.slice, 2], color='blue')
            # ax.view_init(elev=90., azim=180.)
            # ax.set_xlim(ax.get_xlim()[::-1])
            # plt.gca().invert_yaxis()
            # plt.show()

        # f1.close()
        # f2.close()

    def mean_error(self, output_landmarks, landmarks_3D, d):
        mean_error = 0.0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for idx, (output, target) in enumerate(list(zip(output_landmarks, landmarks_3D))):

            output = output.permute(1, 0).to(device)     #[3,68]
            target = target.permute(1, 0).to(device) 

            diff_x = abs(output[0] - target[0])
            diff_y = abs(output[1] - target[1])
            diff_z = abs(output[2] - target[2])

            distances = torch.sqrt(diff_x**2 + diff_y**2 + diff_z**2)
            distances = distances/d[idx]
            sum_distances = torch.sum(distances)
            
        mean_error = sum_distances/68

        return mean_error


    def test(self):

        for epoch in tqdm.trange(self.epoch, self.max_epoch, desc='Train', ncols=80):
            self.epoch = epoch
            self.eval()

        self.writer.close()
