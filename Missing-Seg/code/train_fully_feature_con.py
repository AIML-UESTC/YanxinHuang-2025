import argparse
import logging
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.brats2019 import (BraTS2019, CenterCrop, RandomCrop,
                                   RandomRotFlip, ToTensor,RandomRotFlip_multi,RandomCrop_multi,ToTensor_multi,
                                   TwoStreamBatchSampler,BraTS2019_consis)
from networks.net_factory_3d import net_factory_3d
from utils import losses, metrics, ramps
from val_3D import test_all_case,test_all_case_multi
import os
import copy

parser = argparse.ArgumentParser()
# parser.add_argument('--root_path', type=str,
#                     default='../data/BraTS2019', help='Name of Experiment')
# parser.add_argument('--root_path', type=str,
#                     default='/home/qitam/sdb2/home/qiteam_project/huang/SSL4MIS-master/data/BraTS2019/Dataset303_Flair', help='Name of Experiment')

parser.add_argument('--root_path', type=str,
                    default='/home/qitam/sdb2/home/qiteam_project/huang/SSL4MIS-master/data/BraTSmulti/BraTS2023-GLITraining', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='feature_consistency_concat', help='experiment_name')
parser.add_argument('--mod', type=str,
                    default='multi_concat', help='experiment_mod')
parser.add_argument('--model', type=str,
                    default='unet_feature_con', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=10000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=2,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[96, 96, 96],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--labeled_num', type=int, default=1000,
                    help='labeled data')

args = parser.parse_args()

def sharpening(P, T):
    numerator = P ** (1 / T)
    denominator = P ** (1 / T) + (1 - P) ** (1 / T)
    result = numerator / denominator
    return result


def train(args, snapshot_path):
    base_lr = args.base_lr
    train_data_path = args.root_path
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    num_classes = 2
    model = net_factory_3d(net_type=args.model, in_chns=1, class_num=num_classes)
    db_train = BraTS2019_consis(base_dir=train_data_path,
                             split='train',
                             num=args.labeled_num,
                             transform=transforms.Compose([
                                 RandomRotFlip_multi(),
                                 RandomCrop_multi(args.patch_size),
                                 ToTensor_multi(),
                             ]),
                             mod = args.mod)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)

    encoders = [copy.deepcopy(model[0]) for _ in range(4)]
    decoder = model[1]
    [encoder.train() for encoder in encoders]
    decoder.train()
    encoder_optimizer = [optim.SGD(encoder.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001) for encoder in encoders]
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(2)
    mse_loss = nn.MSELoss()
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            # 处理成二分类
            label_batch[label_batch != 0] = 1
            feature = {}
            conv4 = {}
            conv3 = {}
            conv2 = {}
            conv1 = {}

            for i in range(4):
                encoded = encoders[i](volume_batch[:,i,:,:,:])
                if i == 0:
                    feature['T1'] = encoded[0]
                    conv4['T1'] = encoded[1]
                    conv3['T1'] = encoded[2]
                    conv2['T1'] = encoded[3]
                    conv1['T1'] = encoded[4]
                elif i == 1:
                    feature['T1ce'] = encoded[0]
                    conv4['T1ce'] = encoded[1]
                    conv3['T1ce'] = encoded[2]
                    conv2['T1ce'] = encoded[3]
                    conv1['T1ce'] = encoded[4]
                elif i == 2:
                    feature['T2'] = encoded[0]
                    conv4['T2'] = encoded[1]
                    conv3['T2'] = encoded[2]
                    conv2['T2'] = encoded[3]
                    conv1['T2'] = encoded[4]
                elif i == 3:
                    feature['T2f'] = encoded[0]
                    conv4['T2f'] = encoded[1]
                    conv3['T2f'] = encoded[2]
                    conv2['T2f'] = encoded[3]
                    conv1['T2f'] = encoded[4]
            concat_feature = torch.cat((feature['T1'],feature['T1ce'],feature['T2'],feature['T2f']),dim=1)
            concat_conv4 = torch.cat((conv4['T1'],conv4['T1ce'],conv4['T2'],conv4['T2f']),dim=1)
            concat_conv3 = torch.cat((conv3['T1'],conv3['T1ce'],conv3['T2'],conv3['T2f']),dim=1)
            concat_conv2 = torch.cat((conv2['T1'],conv2['T1ce'],conv2['T2'],conv2['T2f']),dim=1)
            concat_conv1 = torch.cat((conv1['T1'],conv1['T1ce'],conv1['T2'],conv1['T2f']),dim=1)
            decoded = decoder(concat_feature, concat_conv4, concat_conv3, concat_conv2, concat_conv1)
            decoded_soft = torch.softmax(decoded, dim=1)
            loss_dice = dice_loss(decoded_soft, label_batch.unsqueeze(1))
            loss_ce = ce_loss(decoded_soft, label_batch)
            loss = 0.5 * (loss_dice + loss_ce)

            noise = torch.randn_like(volume_batch)
            noisy_volume_batch = volume_batch+noise
            feature_1 = {}
            conv4_1 = {}
            conv3_1 = {}
            conv2_1 = {}
            conv1_1 = {}
            for i in range(4):
                encoded_1 = encoders[i](noisy_volume_batch[:,i,:,:,:])
                if i == 0:
                    feature_1['T1'] = encoded_1[0]
                    conv4_1['T1'] = encoded_1[1]
                    conv3_1['T1'] = encoded_1[2]
                    conv2_1['T1'] = encoded_1[3]
                    conv1_1['T1'] = encoded_1[4]
                elif i == 1:
                    feature_1['T1ce'] = encoded_1[0]
                    conv4_1['T1ce'] = encoded_1[1]
                    conv3_1['T1ce'] = encoded_1[2]
                    conv2_1['T1ce'] = encoded_1[3]
                    conv1_1['T1ce'] = encoded_1[4]
                elif i == 2:
                    feature_1['T2'] = encoded_1[0]
                    conv4_1['T2'] = encoded_1[1]
                    conv3_1['T2'] = encoded_1[2]
                    conv2_1['T2'] = encoded_1[3]
                    conv1_1['T2'] = encoded_1[4]
                elif i == 3:
                    feature_1['T2f'] = encoded_1[0]
                    conv4_1['T2f'] = encoded_1[1]
                    conv3_1['T2f'] = encoded_1[2]
                    conv2_1['T2f'] = encoded_1[3]
                    conv1_1['T2f'] = encoded_1[4]
            concat_feature_1 = torch.cat((feature_1['T1'],feature_1['T1ce'],feature_1['T2'],feature_1['T2f']),dim=1)
            concat_conv4_1 = torch.cat((conv4_1['T1'],conv4_1['T1ce'],conv4_1['T2'],conv4_1['T2f']),dim=1)
            concat_conv3_1 = torch.cat((conv3_1['T1'],conv3_1['T1ce'],conv3_1['T2'],conv3_1['T2f']),dim=1)
            concat_conv2_1 = torch.cat((conv2_1['T1'],conv2_1['T1ce'],conv2_1['T2'],conv2_1['T2f']),dim=1)
            concat_conv1_1 = torch.cat((conv1_1['T1'],conv1_1['T1ce'],conv1_1['T2'],conv1_1['T2f']),dim=1)
            no_decoded = decoder(concat_feature_1, concat_conv4_1, concat_conv3_1, concat_conv2_1, concat_conv1_1)
            no_decoded_soft = torch.softmax(no_decoded, dim=1)
            con_loss = mse_loss(no_decoded_soft,decoded_soft)
            # soft_T1 = feature['T1']+feature['T2f']-feature['T1ce']
            # soft_T1ce = feature['T1ce']+feature['T2']-feature['T1']
            # con_loss = (mse_loss(soft_T1, feature['T2']) + mse_loss(soft_T1, feature['T2f']) + mse_loss(
            #     soft_T1ce, feature['T2']) + mse_loss(soft_T1ce, feature['T2f']))/4
            # con_loss = (mse_loss(feature['T1'], feature['T2']) + mse_loss(feature['T1'], feature['T2f']) + mse_loss(
            #     feature['T1ce'], feature['T2']) + mse_loss(feature['T1ce'], feature['T2f']))/4
            consistency_loss = loss+con_loss
            # 清零所有编码器和解码器的梯度
            for i in range(4):
                encoder_optimizer[i].zero_grad()
            decoder_optimizer.zero_grad()
            consistency_loss.backward()
            for i in range(4):
                encoder_optimizer[i].step()
            decoder_optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in decoder_optimizer.param_groups:
                param_group['lr_encoder'] = lr_
            for optimizer in encoder_optimizer:
                for param_group in optimizer.param_groups:
                    param_group['lr_decoder'] = lr_
            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            # writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/total_loss', consistency_loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            # logging.info(
            #     'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
            #     (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))
            logging.info(
                'iteration %d : total_loss : %f, loss_ce: %f, loss_dice: %f, loss_mse: %f' %
                (iter_num, consistency_loss.item(), loss_ce.item(), loss_dice.item(),con_loss.item()))
            writer.add_scalar('loss/loss', consistency_loss, iter_num)
            if iter_num % 20 == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                image = decoded_soft[0, 1:2, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label',
                                 grid_image, iter_num)

                image = label_batch[0, :, :, 20:61:10].unsqueeze(
                    0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_label',
                                 grid_image, iter_num)
            if iter_num >2000 and iter_num % 200 == 0 :
                [encoder.eval() for encoder in encoders]
                decoder.eval()
                multi_model = [encoders, decoder]
                avg_metric = test_all_case_multi(
                    multi_model, args.root_path, test_list="val.txt", num_classes=2, patch_size=args.patch_size,
                    stride_xy=64, stride_z=64,mod = args.mod)
                if avg_metric[:, 0].mean() > best_performance:
                    for i in range(len(multi_model[0])):
                        best_performance = avg_metric[:, 0].mean()
                        save_encoder_path = os.path.join(snapshot_path,
                                                      'iter_E_{}_{}_dice_{}.pth'.format(i,
                                                          iter_num, round(best_performance, 4)))
                        save_encoder_best = os.path.join(snapshot_path,
                                                 'E_{}_{}_best_model.pth'.format(i,args.model))
                        torch.save(multi_model[0][i].state_dict(), save_encoder_path)
                        torch.save(multi_model[0][i].state_dict(), save_encoder_best)

                    save_decoder_path = os.path.join(snapshot_path,
                                                  'iter_D_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_decoder_best = os.path.join(snapshot_path,
                                             'D_{}_best_model.pth'.format(args.model))
                    torch.save(multi_model[1].state_dict(), save_decoder_path)
                    torch.save(multi_model[1].state_dict(), save_decoder_best)

                writer.add_scalar('info/val_dice_score',
                                  avg_metric[0, 0], iter_num)
                writer.add_scalar('info/val_hd95',
                                  avg_metric[0, 1], iter_num)
                logging.info(
                    'iteration %d : dice_score : %f hd95 : %f' % (iter_num, avg_metric[0, 0].mean(), avg_metric[0, 1].mean()))
                # model.train()
                [encoder.train() for encoder in encoders]
                decoder.train()

            if iter_num % 2000 == 0:
                for i in range(len(encoders)):
                    save_encoder_path = os.path.join(
                        snapshot_path, 'iter_'+str(i)+'_E_' + str(iter_num) + '.pth')
                    torch.save(encoders[i].state_dict(), save_encoder_path)
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_'+'_D_' + str(iter_num) + '.pth')
                torch.save(decoder.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}/{}".format(args.exp, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
