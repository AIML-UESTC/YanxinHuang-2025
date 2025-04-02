import argparse
import logging
import random
import shutil
import sys
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,4,5'
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from monai import transforms as mon_transforms
from monai.losses.dice import DiceLoss as mon_Diceloss
from dataloaders import utils
from dataloaders.brats_diff import (BraTS2019, CenterCrop, RandomCrop_multi,
                                   RandomRotFlip_multi, ToTensor,RandomRotFlip_label,RandomCrop_label,ToTensor_multi,
                                   TwoStreamBatchSampler)
from dataloaders.brats_diff import (BraTS2019, CenterCrop, RandomCrop,
                                   RandomRotFlip, ToTensor,RandomRotFlip_label,RandomCrop_label,ToTensor_multi,
                                   TwoStreamBatchSampler)
from networks.net_factory_3d import net_factory_3d
from utils import losses, metrics, ramps
from val_guide import test_all_case
import os
parser = argparse.ArgumentParser()
# parser.add_argument('--root_path', type=str,
#                     default='/root/autodl-tmp/BraTS2023-GLITraining', help='Name of Experiment')
parser.add_argument('--root_path', type=str,
                    default='/home/qitam/sdb2/home/qiteam_project/huang/SSL4MIS-master/data/BraTSmulti/BraTS2023-GLITraining', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='2023', help='experiment_name')
parser.add_argument('--mod', type=str,
                    default='multi_concat', help='experiment_mod')
parser.add_argument('--model', type=str,
                    default='unet_3D_multi', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=3000, help='maximum epoch number to train')
parser.add_argument('--un_max_iterations', type=int,
                    default=27000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=1,
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

parser.add_argument('--consistency', type=float,
                    default=0.5, help='consistency')
parser.add_argument('--labeled_bs', type=int,
                    default=1, help='consistency')

parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
parser.add_argument('--mask_ratio', type=float, default=2/3, help='ratio of mask/image')
args = parser.parse_args()
def trans_seg(outputs_soft):
    # new_shape = [1, 3, 96, 96, 96]  # 结果的形状根据你的规则决定
    # seg_data = torch.zeros(new_shape)  # 初始化
    #
    # # 划分通道
    # # 规则：
    # # 1. 第二个通道和第四个通道放一起 (索引 1 和 3)
    # # 2. 第二个、第三个和第四个通道放一起 (索引 1, 2 和 3)
    # # 3. 单独放第四个通道 (索引 3)
    #
    # # 按规则组合通道
    # # 第一部分: (seg_data == 1) | (seg_data == 3)
    # seg1 = (outputs_soft[:, 1, :, :, :] > 0) | (outputs_soft[:, 3, :, :, :] > 0)
    #
    # # 第二部分: (seg_data == 1) | (seg_data == 3) | (seg_data == 2)
    # seg2 = (outputs_soft[:, 1, :, :, :] > 0) | (outputs_soft[:, 2, :, :, :] > 0) | (
    #         outputs_soft[:, 3, :, :, :] > 0)
    #
    # # 第三部分: seg_data == 3
    # seg3 = (outputs_soft[:, 3, :, :, :] > 0)
    #
    # # 将结果赋值到新 tensor 中
    # seg_data[:, 0, :, :, :] = seg1.float()  # 将第一个部分结果赋值给新 tensor 的第一个通道
    # seg_data[:, 1, :, :, :] = seg2.float()  # 将第二个部分结果赋值给新 tensor 的第二个通道
    # seg_data[:, 2, :, :, :] = seg3.float()  # 将第三个部分结果赋值给新 tensor 的第三个通道
    new_shape = [1, 3, 96, 96, 96]  # 直接使用输入的形状
    # seg_data = torch.zeros(new_shape, device=outputs_soft.device)  # 初始化

    seg1 = ((outputs_soft[:, 1, :, :, :] > 0) | (outputs_soft[:, 3, :, :, :] > 0)).float()
    seg2 = ((outputs_soft[:, 1, :, :, :] > 0) | (outputs_soft[:, 2, :, :, :] > 0) |
            (outputs_soft[:, 3, :, :, :] > 0)).float()
    seg3 = (outputs_soft[:, 3, :, :, :] > 0).float()

    seg_data = torch.empty(new_shape, device=outputs_soft.device)
    # seg_data.grad = outputs_soft.grad.clone()
    seg_data[:, 0, :, :, :] = seg1  # 第一个部分
    seg_data[:, 1, :, :, :] = seg2  # 第二个部分
    seg_data[:, 2, :, :, :] = seg3

    return seg_data

def assert_shape(test, reference):

    assert test.shape == reference.shape, "Shape mismatch: {} and {}".format(
        test.shape, reference.shape)
class ConfusionMatrix:

    def __init__(self, test=None, reference=None):

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.reference_empty = None
        self.reference_full = None
        self.test_empty = None
        self.test_full = None
        self.set_reference(reference)
        self.set_test(test)

    def set_test(self, test):

        self.test = test
        self.reset()

    def set_reference(self, reference):

        self.reference = reference
        self.reset()

    def reset(self):

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.test_empty = None
        self.test_full = None
        self.reference_empty = None
        self.reference_full = None

    def compute(self):

        if self.test is None or self.reference is None:
            raise ValueError("'test' and 'reference' must both be set to compute confusion matrix.")

        assert_shape(self.test, self.reference)

        self.tp = int(((self.test != 0) * (self.reference != 0)).sum())
        self.fp = int(((self.test != 0) * (self.reference == 0)).sum())
        self.tn = int(((self.test == 0) * (self.reference == 0)).sum())
        self.fn = int(((self.test == 0) * (self.reference != 0)).sum())
        self.size = int(np.prod(self.reference.shape, dtype=np.int64))
        self.test_empty = not np.any(self.test)
        self.test_full = np.all(self.test)
        self.reference_empty = not np.any(self.reference)
        self.reference_full = np.all(self.reference)

    def get_matrix(self):

        for entry in (self.tp, self.fp, self.tn, self.fn):
            if entry is None:
                self.compute()
                break

        return self.tp, self.fp, self.tn, self.fn

    def get_size(self):

        if self.size is None:
            self.compute()
        return self.size

    def get_existence(self):

        for case in (self.test_empty, self.test_full, self.reference_empty, self.reference_full):
            if case is None:
                self.compute()
                break

        return self.test_empty, self.test_full, self.reference_empty, self.reference_full


def dice(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """2TP / (2TP + FP + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty and reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(2. * tp / (2 * tp + fp + fn))
def validation_step(self, batch):
    image, label = self.get_input(batch)

    output = self.window_infer(image, self.model, pred_type="ddim_sample")

    output = torch.sigmoid(output)

    output = (output > 0.5).float().cpu().numpy()

    target = label.cpu().numpy()
    o = output[:, 1]
    t = target[:, 1]  # ce
    wt = dice(o, t)
    # core
    o = output[:, 0]
    t = target[:, 0]
    tc = dice(o, t)
    # active
    o = output[:, 2]
    t = target[:, 2]
    et = dice(o, t)

    return [wt, tc, et]
def fill_missing_channel(volume_batch, label_batch):
    # 创建与 volume_batch 相同形状的输出张量
    output = volume_batch.clone()

    # 提取 T1, T1ce 和 T2w（要填充的）通道
    T1 = volume_batch[:, 0, :, :, :]
    T1ce = volume_batch[:, 1, :, :, :]
    T2w_filled = volume_batch[:, 3, :, :, :]

    # # 确定病灶区域：根据 label_batch 中非零区域确定
    # lesion_mask = (label_batch > 0)
    # 定义立方体范围 [0:96, 0:96, 0:96]
    cube_range = slice(0, 96)

    # 病灶区域掩码（限制在立方体范围内）
    lesion_mask = (label_batch[:, cube_range, cube_range, cube_range] > 0)

    # 将 T2w_filled 的病灶区域用 T1ce 填充，其他区域用 T1 填充
    T2w_filled[:, cube_range, cube_range, cube_range][lesion_mask] = \
        T1ce[:, cube_range, cube_range, cube_range][lesion_mask]
    T2w_filled[:, cube_range, cube_range, cube_range][~lesion_mask] = \
        T1[:, cube_range, cube_range, cube_range][~lesion_mask]

    # # 将 T2w_filled 的病灶区域用 T1ce 填充，其他区域用 T1 填充
    # T2w_filled[lesion_mask] = T1ce[lesion_mask]
    # T2w_filled[~lesion_mask] = T1[~lesion_mask]

    # 更新输出张量的第 3 通道
    output[:, 3, :, :, :] = T2w_filled

    return output
def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)
def train(args, snapshot_path):
    base_lr = args.base_lr
    train_data_path = args.root_path
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    un_iter = args.un_max_iterations
    num_classes = 4

    def create_model(ema=False,guide = False,args = None):
        # Network definition
        if guide:
            # net = net_factory_3d(net_type='diff', in_chns=1, class_num=num_classes,args=args)
            net = net_factory_3d(net_type='diff', in_chns=1, class_num=num_classes,args=args)
            model = net.cuda()
        else:
            net = net_factory_3d(net_type=args.model, in_chns=1, class_num=num_classes)
            model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model
    if 'multi' in args.mod:
        guide_model = create_model(guide=True,args=args)
        model = create_model(args=args)
        # turb_model = create_model(ema=False)
        # model = net_factory_3d(net_type=args.model, in_chns=1, class_num=num_classes)

        db_train_sup = BraTS2019(base_dir=train_data_path,
                             split='train',
                             num=args.labeled_num,
                             sup = 'sup',
                             transform=transforms.Compose([
                                 RandomRotFlip_multi(),
                                 RandomCrop_multi(args.patch_size),
                                 ToTensor_multi(),
                             ]),
                             mod = args.mod)
        db_train_un = BraTS2019(base_dir=train_data_path,
                             split='train',
                             num=args.labeled_num,
                             sup = 'un',
                             transform=transforms.Compose([
                                 RandomRotFlip_multi(),
                                 RandomCrop_multi(args.patch_size),
                                 ToTensor_multi(),
                             ]),
                             mod = args.mod)
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    # 加载预训练的模型权重
    # guide_pretrained_model_path = '/root/autodl-tmp/model/2023_diff/diff/iter_23800_dice_0.6183.pth'
    guide_pretrained_model_path = '/home/qitam/sdb2/home/qiteam_project/huang/SSL4MIS-master/model/2023_diff/diff/iter_12000_dice_0.5738.pth'
    # pretrained_model_path = '/root/SSL4MIS-master/model/2023_semi_10_1/unet_3D_multi/iter_3000.pth'
    # base,extension = os.path.splitext(pretrained_model_path)
    guide_model.load_state_dict(torch.load(guide_pretrained_model_path))
    # model.load_state_dict(torch.load(pretrained_model_path))

    supervised_loader = DataLoader(db_train_sup, batch_size=1, shuffle=True,worker_init_fn = worker_init_fn)
    unsupervised_loader = DataLoader(db_train_un, batch_size=1, shuffle=True)
    model.train()
    # guide_model.train()
    # turb_model.train()
    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    optimizer_guide = optim.SGD(guide_model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(4)
    ai_dice = mon_Diceloss(sigmoid=True,include_background=False)
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    writer = SummaryWriter(snapshot_path + '/log')
    # logging.info("{} iterations per epoch".format(len(trainloader)))
    logging.info("{} iterations per epoch".format(len(supervised_loader)))
    # if pretrained_model_path is not None:
    #     iter_num = int(os.path.basename(base).split('_')[1])
    # else:
    #     iter_num = 0
    iter_num = 0
    # max_epoch = max_iterations // len(trainloader) + 1
    # max_epoch = (max_iterations-iter_num) // len(trainloader) + 1
    max_sup_epoch = (max_iterations-iter_num) // len(supervised_loader)
    best_performance = 0.0
    max_un_epoch = (un_iter-iter_num) // len(unsupervised_loader)
    iterator_sup = tqdm(range(max_sup_epoch), ncols=70)
    iterator_un = tqdm(range(max_un_epoch), ncols=70)
    for epoch_num in iterator_sup:
        # for i_batch, sampled_batch in enumerate(trainloader):
        for i_batch, sampled_batch in enumerate(supervised_loader):
            print(sampled_batch['image_name'])
            total_batch, to_label_batch = sampled_batch['image'], sampled_batch['label']
            total_batch, to_label_batch = total_batch.cuda(), to_label_batch.cuda()

            volume_batch = total_batch[0:args.labeled_bs, :, :, :, :]
            # unlabeled_batch = total_batch[args.labeled_bs:, :, :, :, :]

            label_batch = to_label_batch[0:args.labeled_bs, :, :, :, :]
            reconstructed_seg_data = torch.zeros_like(label_batch[:, 0], dtype=torch.int32, device=label_batch.device)

            # 恢复 seg_data 的标签
            reconstructed_seg_data[label_batch[:, 2].bool()] = 3  # 恢复标签3的部分
            reconstructed_seg_data[label_batch[:, 0].bool() & ~label_batch[:, 2].bool()] = 1  # 恢复标签1的部分
            reconstructed_seg_data[label_batch[:, 1].bool() & ~label_batch[:, 0].bool()] = 2

            if i_batch%2==0:
                volume_batch[:,3,:,:,:]=0
                # volume_batch = fill_missing_channel(volume_batch, label_batch)
                volume_batch = fill_missing_channel(volume_batch, reconstructed_seg_data)
                print('volume_batch')
            # diffusion
            image = volume_batch
            # unlabeled_image = unlabeled_batch
            x_start = label_batch.float()

            # pred_unlabeled = guide_model(image=unlabeled_image, pred_type="ddim_sample")
            #
            x_start = (x_start) * 2 - 1
            x_t, t, noise = guide_model(x=x_start, pred_type="q_sample")
            # pred_xstart = guide_model(x=x_t, step=t, image=image, pred_type="denoise")
            # x_t, t, noise = guide_model(x=low_freq_label, pred_type="q_sample")
            pred_xstart = guide_model(x=x_t, step=t, image=image, pred_type="denoise")
            # pred_unlabeled = guide_model(image=unlabeled_image, pred_type="ddim_sample")# diff的无监督输出
            pred_xstart = (pred_xstart>0.5).float()

            # pred_unlabeled = (pred_unlabeled>0.5).float()
            loss_dice = ai_dice(pred_xstart, label_batch)
            loss_bce = bce(pred_xstart, label_batch)

            pred_xstart = torch.sigmoid(pred_xstart)
            loss_mse = mse(pred_xstart, label_batch)

            #Unet

            # outputs = model(volume_batch,high_freq_label)
            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs,dim=1)

            # un_out = model(unlabeled_batch)
            # un_out_soft = torch.softmax(un_out, dim=1)# Unet的无监督输出

            unet_ce_loss = ce_loss(outputs_soft,reconstructed_seg_data.long())
            unet_dice_loss = dice_loss(outputs_soft,reconstructed_seg_data.unsqueeze(0))
            unet_loss = 0.5*(unet_ce_loss + unet_dice_loss)
            # unet_dice_loss = ai_dice(outputs_soft, label_batch)
            tran_seg = trans_seg(outputs_soft)
            # tran_un_out = trans_seg(un_out_soft)

            consistency_weight = get_current_consistency_weight(iter_num // 150)
            # con_loss = ai_dice(pred_xstart, tran_seg.cuda())
            con_loss = torch.mean((pred_xstart - tran_seg.cuda()) ** 2)
            # un_con_loss = torch.mean((pred_unlabeled - tran_un_out.cuda()) ** 2)
            # loss = loss_dice + loss_bce + loss_mse+unet_dice_loss+con_loss
            # loss = loss_dice + loss_bce + loss_mse+unet_loss+consistency_weight*con_loss + consistency_weight*un_con_loss
            loss = loss_dice + loss_bce + loss_mse+unet_loss+consistency_weight*con_loss
            optimizer.zero_grad()
            # optimizer_guide.zero_grad()
            loss.backward()
            optimizer.step()
            # optimizer_guide.step()
            logging.info(
                'iteration %d : loss : %f, loss_dice: %f, loss_bce: %f,loss_mse: %f ,unet_loss: %f,unet_ce_loss: %f,unet_dice_loss: %f,unet_con_loss: %f,consistency_weight: %f' %
                (iter_num, loss.item(), loss_dice.item(), loss_bce.item(),loss_mse.item(),unet_loss.item(),unet_ce_loss.item(),unet_dice_loss.item(),consistency_weight*con_loss.item(),consistency_weight))
            writer.add_scalar('loss/loss', loss, iter_num)
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            if iter_num % 100 == 0:
                non_zero_slices = torch.nonzero(reconstructed_seg_data.squeeze(0).sum(dim=(0, 1)))
                if len(non_zero_slices)>=5:
                    middle_idx = len(non_zero_slices) // 2  # 中间位置的索引
                    start_idx = max(0, middle_idx - 5)
                    end_idx = min(len(non_zero_slices), middle_idx + 6)
                    selected_slices = non_zero_slices[start_idx:end_idx].squeeze()
                # elif non_zero_slices.size(0) != 0:
                #     selected_slices = non_zero_slices
                else:
                    selected_slices = [20,30,40,50,60]

                image = volume_batch[0, 0:1, :, :, selected_slices].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                image = pred_xstart[0, 0:1, :, :, selected_slices].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_wt_label',
                                 grid_image, iter_num)
                image = pred_xstart[0, 1:2, :, :, selected_slices].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_tc_label',
                                 grid_image, iter_num)
                image = pred_xstart[0, 2:3, :, :, selected_slices].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_et_label',
                                 grid_image, iter_num)

                image = reconstructed_seg_data[0, :, :, selected_slices].unsqueeze(0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_label',
                                 grid_image, iter_num)

            if iter_num> 1000 and iter_num % 200 == 0:
            # if iter_num> 0 and iter_num % 1 == 0:
                model.eval()
                # avg_metric = np.array([1.0, 1.0, 1.0])
                avg_metric = test_all_case(
                    model, args.root_path, test_list="val.txt", num_classes=num_classes, patch_size=args.patch_size,
                    stride_xy=64, stride_z=64,mod = args.mod)
                if avg_metric[:, 0].mean() > best_performance:
                    best_performance = avg_metric[:, 0].mean()
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                writer.add_scalar('info/val_dice_score',
                                  avg_metric[:, 0].mean(), iter_num)
                writer.add_scalar('info/val_hd95',
                                  avg_metric[:, 1].mean(), iter_num)
                logging.info(
                    'iteration %d : dice_score : %f hd95 : %f' % (iter_num, avg_metric[:, 0].mean(), avg_metric[:, 1].mean()))
                model.train()

            if iter_num> 2000 and iter_num % 200==0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

        # if iter_num >= max_iterations:
        #         break
        # if iter_num >= max_iterations:
        #     iterator_sup.close()
        #     break

    un_iter_num = iter_num
    for epoch_num in iterator_un:
        for i_batch, sampled_batch in enumerate(unsupervised_loader):
            print('开始无监督训练',iter_num)
            print(sampled_batch['image_name'])
            total_batch, to_label_batch = sampled_batch['image'], sampled_batch['label']
            total_batch, to_label_batch = total_batch.cuda(), to_label_batch.cuda()

            unlabeled_batch = total_batch
            # diffusion
            unlabeled_image = unlabeled_batch

            pred_unlabeled = guide_model(image=unlabeled_image, pred_type="ddim_sample")
            pred_unlabeled = (pred_unlabeled > 0.5).float()

            un_sup_seg = torch.zeros((1, 4, 96, 96, 96), dtype=torch.long).cuda()  # 创建全零张量

            # 根据取值将标签填入
            un_sup_seg[0, 1][(pred_unlabeled[0, 0].bool() & ~pred_unlabeled[0, 2].bool())] = 1  # 标签1
            un_sup_seg[0, 2][pred_unlabeled[0, 1].bool() & ~pred_unlabeled[0, 0].bool()] = 1  # 标签2
            un_sup_seg[0, 3][pred_unlabeled[0, 2].bool()] = 1
            # Unet

            un_out = model(unlabeled_batch)
            un_out_soft = torch.softmax(un_out, dim=1)  # Unet的无监督输出
            # tran_un_out = trans_seg(un_out_soft)

            consistency_weight_1 = get_current_consistency_weight(iter_num // 150)
            un_con_loss = torch.mean((un_sup_seg - un_out_soft.cuda()) ** 2)
            # unloss = consistency_weight_1 * un_con_loss
            unloss = un_con_loss
            optimizer.zero_grad()
            unloss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            un_iter_num = un_iter_num + 1
            writer.add_scalar('info/lr', lr_, un_iter_num)
            writer.add_scalar('info/total_loss', unloss, un_iter_num)
            writer.add_scalar('info/con_weight', consistency_weight_1, un_iter_num)

            logging.info(
                'iteration %d : unloss : %f consistency_weight_1 :%f' %
                (un_iter_num, unloss.item(), consistency_weight_1))
            if un_iter_num % 2000 == 0:
                non_zero_slices = torch.nonzero(un_sup_seg.squeeze(0).sum(dim=(0,1, 2)))
                if len(non_zero_slices) >= 5:
                    middle_idx = len(non_zero_slices) // 2  # 中间位置的索引
                    start_idx = max(0, middle_idx - 5)
                    end_idx = min(len(non_zero_slices), middle_idx + 6)
                    selected_slices = non_zero_slices[start_idx:end_idx].squeeze()
                # elif non_zero_slices.size(0) != 0:
                #     selected_slices = non_zero_slices
                else:
                    selected_slices = [20, 30, 40, 50, 60]

                image = total_batch[0, 0:1, :, :, selected_slices].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, un_iter_num)

                image = un_sup_seg[0, 0:1, :, :, selected_slices].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_wt_label',
                                 grid_image, un_iter_num)
                image = un_sup_seg[0, 1:2, :, :, selected_slices].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_tc_label',
                                 grid_image, un_iter_num)
                image = un_sup_seg[0, 2:3, :, :, selected_slices].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_et_label',
                                 grid_image, un_iter_num)

                image = to_label_batch[0, 1:2,:, :, selected_slices].permute(3, 0, 1, 2).repeat(1, 3,1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_wt_label',
                                 grid_image, un_iter_num)

            if un_iter_num > 10000 and un_iter_num % 200 == 0:
                # if iter_num> 0 and iter_num % 1 == 0:
                model.eval()
                # avg_metric = np.array([1.0, 1.0, 1.0])
                avg_metric = test_all_case(
                    model, args.root_path, test_list="val.txt", num_classes=num_classes, patch_size=args.patch_size,
                    stride_xy=64, stride_z=64, mod=args.mod)
                if avg_metric[:, 0].mean() > best_performance:
                    best_performance = avg_metric[:, 0].mean()
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      un_iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                writer.add_scalar('info/val_dice_score',
                                  avg_metric[:, 0].mean(), un_iter_num)
                writer.add_scalar('info/val_hd95',
                                  avg_metric[:, 1].mean(), un_iter_num)
                logging.info(
                    'iteration %d : dice_score : %f hd95 : %f' % (
                    un_iter_num, avg_metric[:, 0].mean(), avg_metric[:, 1].mean()))
                model.train()

            if un_iter_num > 10000 and un_iter_num % 2000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(un_iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
        if iter_num >= max_iterations+un_iter:
                break
        if iter_num >= max_iterations+un_iter:
            iterator_un.close()
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
