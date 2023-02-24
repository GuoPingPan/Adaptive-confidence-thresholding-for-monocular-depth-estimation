## server 17 parameter variance version --> best

from __future__ import print_function
import argparse
#from DataLoader_wlidar2_pretrained import *
from DataLoader_wlidar2_uncertaintyPAC import *
from model_w_uncertainyPAC import *
from datetime import datetime
import os
import sys
import numpy as np
import time
from skimage import io
import util
import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader
from torch.autograd import Variable

# Constants
NUM_INPUT_CHANNELS = 3
NUM_EPOCHS = 50

# learning_rate = 1e-5
MOMENTUM = 0.9
BATCH_SIZE = 8

h = 192
w = 480

#####dir_checkpoint1 = '/home/cvlab/Documents/pytorch-segnet-master/src/model/200330_train_w_depth/epoch005_model.pth'
dir_checkpoint1 = '/home/cvlab/Documents/pytorch-segnet-master/src/model_param/201114_unctrain/epoch031_param_model1.pth'
dir_checkpoint2 = '/home/cvlab/Documents/pytorch-segnet-master/src/model_param/201104_noconcat/epoch050_param_model2.pth'
dir_checkpoint3 = '/home/cvlab/Documents/CCNN_korea/CCNN_confidence/Code/model/200820_wParamtrain/K12_ganet_CCNN_458_confmodel.pth'
#####dir_checkpoint5 = '/home/cvlab/Documents/pytorch-segnet-master/src/model_param/201114_unctrain/epoch030_param_model5.pth'
dir_checkpoint5 = '/home/cvlab/Documents/pytorch-segnet-master/src/model_param/201114_unctrain/epoch031_param_model5.pth'

# Arguments
parser = argparse.ArgumentParser(description='Train a model with parameter network')
parser.add_argument('--save_dir', default='/home/cvlab/Documents/pytorch-segnet-master/src/model_param/210120_LuwithC')
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--min_depth', type=float, help='minimum depth for evaluation', default=1e-3)
parser.add_argument('--max_depth', type=float, help='maximum depth for evaluation', default=80)
parser.add_argument('--factor', type=int, default=8, metavar='R',
                        help='upsampling factor')
args = parser.parse_args()

initialized = True


# L1loss = torch.nn.L1Loss(reduction='sum')

def train():
    is_better = True
    # accuracy = 0
    prev_loss = float('inf')
    #torch.autograd.set_detect_anomaly(True)

    model1.train()
    model5.train()
    BCELoss = nn.BCELoss()

    for epoch in range(NUM_EPOCHS):
        loss_f = 0

        learning_rate = 1e-4
        learning_rate = learning_rate * (0.1 ** (epoch // 30))

        optimizer1 = torch.optim.Adam(model1.parameters(), lr=learning_rate)
        optimizer5 = torch.optim.Adam(model5.parameters(), lr=learning_rate)

        for batch in train_dataloader:
            input_tensor = Variable(batch['left_img'])
            disp_tensor = Variable(batch['disp_img'])
            disparity_tensor = Variable(batch['disparity_img'])

            if CUDA:
                input_tensor = input_tensor.cuda()
                disp_tensor = disp_tensor.cuda()
                disparity_tensor = disparity_tensor.cuda()

            #conduct with left img
            predicted_tensor, tempact, tempindice, pacresult = model1(input_tensor)
            #temppred, activation, indiceslist = model4(input_tensor)
            uncertain_input = tempact[4]
            uncertain_tensor = model5(uncertain_input, tempindice)
            conf_tensor = model3(disparity_tensor)
            th = model2(conf_tensor)

            size = predicted_tensor.shape[0]

            #print('th : ', th)
            #print('rth : ', rth)

            epsilon = -10

            conf_mask = 1 / (1 + torch.exp(epsilon * (conf_tensor - th.view((th.shape[0], 1, 1, 1)))))

            optimizer1.zero_grad()
            optimizer5.zero_grad()

            pred_d = predicted_tensor
            gt_d = disp_tensor

            k_un = 1


            pred_d_final = (torch.exp(-uncertain_tensor.clone()/k_un) * predicted_tensor.clone()) + ((1 - torch.exp(-uncertain_tensor.clone()/k_un)) * pacresult)
            #pred_d_final = (predicted_tensor.clone() + pacresult) / 2
            #pred_d_final = torch.exp(-uncertain_tensor) * predicted_tensor# + ((1 - torch.exp(-uncertain_tensor / k_un)) * pacresult)

            pred_d[pred_d < args.min_depth] = args.min_depth
            pred_d[pred_d > args.max_depth] = args.max_depth
            cap_mask = ((gt_d > args.min_depth) & (gt_d < args.max_depth)).type(torch.float32)
            mask = conf_mask * cap_mask.cuda()

            pred_d_final[pred_d_final < args.min_depth] = args.min_depth
            pred_d_final[pred_d_final > args.max_depth] = args.max_depth
            cap_mask_final = ((gt_d > args.min_depth) & (gt_d < args.max_depth)).type(torch.float32)
            mask_final = conf_mask * cap_mask_final.cuda()

            k = 1e-3

            loss_dc = (torch.sum(torch.abs(pred_d - gt_d) * mask) / torch.sum(mask)) / 80
            loss_dc_final = (torch.sum(torch.abs(pred_d_final - gt_d) * mask_final) / torch.sum(mask_final)) / 80

            #loss_u = torch.mean(torch.abs(pred_d - gt_d) / uncertain_tensor + torch.log(uncertain_tensor))
            loss_u = torch.mean(torch.abs(pred_d - gt_d) * mask / uncertain_tensor + torch.log(uncertain_tensor))

            loss = loss_dc + loss_dc_final + k * loss_u
            print('loss_dc : ', loss_dc)
            print('loss_dc_final : ', loss_dc_final)
            print('loss_u : ', k * loss_u)

            loss.mean().backward()

            optimizer1.step()
            optimizer5.step()

            loss_f += loss.float()

        torch.save(model1.state_dict(), os.path.join(args.save_dir, "epoch%.3d_param_model1.pth" % (epoch + 1)))
        torch.save(model5.state_dict(), os.path.join(args.save_dir, "epoch%.3d_param_model5.pth" % (epoch + 1)))

        print("Epoch #{epoch}\tTrain_Loss: {loss}".format(epoch=epoch + 1, loss=loss_f.item()))
        # print("Epoch #{epoch}\tLoss: {loss}\t Accuracy: {acc}s".format(epoch=epoch + 1, loss=loss_f.item(), acc=final_acc))

        input_numpy = util.tensor2im(input_tensor[0, :, :, :])
        img_dir = "/home/cvlab/Documents/pytorch-segnet-master/src/depth_param/image/210120_LuwithC"
        img_path = os.path.join(img_dir, 'epoch%.3d_train_input.jpg' % (epoch + 1))
        # input_numpy = input_tensor[0].permute(1,2,0).detach().cpu().numpy()
        util.save_image2_(input_numpy, img_path)

        result_dir = "/home/cvlab/Documents/pytorch-segnet-master/src/depth_param/image/210120_LuwithC"
        result_path = os.path.join(result_dir, 'epoch%.3d_train_disparity.jpg' % (epoch + 1))
        result_numpy = predicted_tensor[0].detach().cpu().numpy()
        util.save_image_(result_numpy, result_path)

        final_path = os.path.join(result_dir, 'epoch%.3d_train_disparity_final.jpg' % (epoch + 1))
        final_numpy = pred_d_final[0].detach().cpu().numpy()
        util.save_image_(final_numpy, final_path)

        gt_path = os.path.join(result_dir, 'epoch%.3d_train_gt.jpg' % (epoch + 1))
        gt_numpy = disp_tensor[0].detach().cpu().numpy()
        util.save_image_(gt_numpy, gt_path)

        conf_path = os.path.join(result_dir, 'epoch%.3d_train_conf.jpg' % (epoch + 1))
        conf_numpy = conf_mask[0][0].detach().cpu().numpy()
        io.imsave(conf_path, conf_numpy)

        uncertain_path = os.path.join(result_dir, 'epoch%.3d_train_uncertain.jpg' % (epoch + 1))
        uncertain_numpy = uncertain_tensor[0][0].detach().cpu().numpy()
        io.imsave(uncertain_path, uncertain_numpy)

        """with torch.no_grad():
            model1.eval()
            model2.eval()

            for batch in val_dataloader:
                num += 1
                input_tensor_v = Variable(batch['left_img']).cuda()
                disp_tensor_v = Variable(batch['disp_img']).cuda()

                predicted_tensor_v, activation = model1(input_tensor_v)

                size = predicted_tensor_v.shape[0]
                predicted_tensor_v = predicted_tensor_v.view((size,h,w))

                pred = predicted_tensor_v.detach().cpu().numpy()
                gt = disp_tensor_v.detach().cpu().numpy()

                thresh = np.maximum((gt / pred), (pred / gt))
                a1 = (thresh < 1.25).mean()
                sum_a1 += a1

            print("Epoch #{epoch}\tVal_acc: {acc}\t".format(epoch=epoch+1, acc=sum_a1/num))"""


if __name__ == "__main__":

    CUDA = args.gpu is not None

    train_dataset = KITTIDataloader()

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=32)

    if CUDA:
        model1 = SegNet(input_channels=NUM_INPUT_CHANNELS,
                        output_channels=1).cuda()
        model2 = ParamNet2().cuda()
        model3 = CCNN().cuda()
        #model4 = SegEncoder(input_channels=NUM_INPUT_CHANNELS,
                            #output_channels=1).cuda()
        model5 = UncertainNet().cuda()

        model1 = nn.DataParallel(model1)
        model2 = nn.DataParallel(model2)
        model3 = nn.DataParallel(model3)
        #model4 = nn.DataParallel(model4)
        model5 = nn.DataParallel(model5)

        model1.load_state_dict(torch.load(dir_checkpoint1), strict=False)
        model2.load_state_dict(torch.load(dir_checkpoint2))
        model3.load_state_dict(torch.load(dir_checkpoint3))
        #model4.load_state_dict(torch.load(dir_checkpoint4))
        #model5.load_state_dict(torch.load(dir_checkpoint5))

        for para in model2.parameters():
            para.requires_grad = False

        for para in model3.parameters():
            para.requires_grad = False

        #for para in model4.parameters():
            #para.requires_grad = False

        #for para in model5.parameters():
            #para.requires_grad = False

    train()