import argparse
import logging
import os
import random
import shutil
import sys
import time
from skimage.measure import label
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
                                   TwoStreamBatchSampler)
# from dataloaders.brats_37 import (BraTS2019, CenterCrop, RandomCrop,
#                                    RandomRotFlip, ToTensor,RandomRotFlip_multi,RandomCrop_multi,ToTensor_multi,
#                                    TwoStreamBatchSampler)
from networks.net_factory_3d import net_factory_3d
from utils import losses, metrics, ramps
# from val_3D import test_all_case
from val_turb import test_all_case
import os
import cp_losses
parser = argparse.ArgumentParser()
# parser.add_argument('--root_path', type=str,
#                     default='/home/qitam/sdb2/home/qiteam_project/huang/SSL4MIS-master/data/BraTSmulti/BraTS2023-GLITraining', help='Name of Experiment')

parser.add_argument('--root_path', type=str,
                    default='/root/autodl-tmp/BraTS2023-GLITraining', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='2023_T2f_supply_by_cp', help='experiment_name')
parser.add_argument('--mod', type=str,
                    default='multi_concat', help='experiment_mod')
parser.add_argument('--model', type=str,
                    default='unet_3D_multi', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
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

parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
parser.add_argument('--mask_ratio', type=float, default=2/3, help='ratio of mask/image')
args = parser.parse_args()
def fill_missing_channel(volume_batch, label_batch):
    # 创建与 volume_batch 相同形状的输出张量
    output = volume_batch.clone()

    # 提取 T1, T1ce 和 T2w（要填充的）通道
    T1 = volume_batch[:, 0, :, :, :]
    T1ce = volume_batch[:, 1, :, :, :]
    T2w_filled = volume_batch[:, 3, :, :, :]

    # 确定病灶区域：根据 label_batch 中非零区域确定
    lesion_mask = (label_batch > 0)

    # 将 T2w_filled 的病灶区域用 T1ce 填充，其他区域用 T1 填充
    T2w_filled[lesion_mask] = T1ce[lesion_mask]
    T2w_filled[~lesion_mask] = T1[~lesion_mask]

    # 更新输出张量的第 3 通道
    output[:, 3, :, :, :] = T2w_filled

    return output
def mix_loss(output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    dice_loss = cp_losses.DiceLoss(n_classes=2)
    CE = nn.CrossEntropyLoss(reduction='none')
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    output_soft = F.softmax(output, dim=1)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask
    loss_dice = dice_loss(output_soft, img_l.unsqueeze(1), mask.unsqueeze(1)) * image_weight
    loss_dice += dice_loss(output_soft, patch_l.unsqueeze(1), patch_mask.unsqueeze(1)) * patch_weight
    loss_ce = image_weight * (CE(output, img_l) * mask).sum() / (mask.sum() + 1e-16)
    loss_ce += patch_weight * (CE(output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)#loss = loss_ce
    return loss_dice, loss_ce
class DiceLoss_1(nn.Module):
    # def __init__(self, n_classes):
    #     self.n_classes = n_classes
    def __init__(self, n_classes):
        super(DiceLoss_1, self).__init__()
        self.n_classes = n_classes
    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def dice_coefficient(self, pred, target, smooth=1e-6):
        pred = pred.float()
        target = target.float()
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2. * intersection + smooth) / (union + smooth)
        return dice

    def dice_loss_1(self, pred, target, smooth=1e-6):
        dice = self.dice_coefficient(pred, target, smooth)
        return 1 - dice

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        # target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(self.n_classes):
            dice = self.dice_loss_1(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def get_copy_taste(input,label):
    output = input.clone()
    nonzero = torch.nonzero(label[0,:,:,:])
    if len(nonzero) == 0:
        pass
    else:
        x_min = nonzero[:, 0].min().item()
        x_max = nonzero[:, 0].max().item()
        y_min = nonzero[:, 1].min().item()
        y_max = nonzero[:, 1].max().item()
        z_min = nonzero[:, 2].min().item()
        z_max = nonzero[:, 2].max().item()
        cpoy_part_T1 = input[0,0,:,:,:][x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
        cpoy_part_T1ce = input[0,1,:,:,:][x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
        cpoy_part_T2w = input[0,2,:,:,:][x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
        cpoy_part_T2f = input[0,3,:,:,:][x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]

        temp_T1 = cpoy_part_T1.clone()
        cpoy_part_T1 = cpoy_part_T2w
        cpoy_part_T2w = temp_T1

        temp_T1ce = cpoy_part_T1ce.clone()
        cpoy_part_T1ce = cpoy_part_T2f
        cpoy_part_T2f = temp_T1ce


        # 将互换后的部分放回原图像的对应位置
        output[0, 0, x_min:x_max + 1, y_min:y_max + 1, z_min:z_max + 1] = cpoy_part_T1
        output[0, 2, x_min:x_max + 1, y_min:y_max + 1, z_min:z_max + 1] = cpoy_part_T2w

        output[0, 1, x_min:x_max + 1, y_min:y_max + 1, z_min:z_max + 1] = cpoy_part_T1ce
        output[0, 3, x_min:x_max + 1, y_min:y_max + 1, z_min:z_max + 1] = cpoy_part_T2f

    return output
def bidierction(input,label,img_mask):

    T1 = input[0,0,:,:,:]
    T1ce = input[0,1,:,:,:]
    T2 = input[0,2,:,:,:]
    T2f = input[0,3,:,:,:]

    mix_T1 = (T1 * img_mask + T1ce * (1 - img_mask)).unsqueeze(0).unsqueeze(0)
    mix_T1ce = (T1ce * img_mask + T1 * (1 - img_mask)).unsqueeze(0).unsqueeze(0)
    mix_T2 = (T2 * img_mask + T2f * (1 - img_mask)).unsqueeze(0).unsqueeze(0)
    mix_T2f = (T2f * img_mask + T2 * (1 - img_mask)).unsqueeze(0).unsqueeze(0)

    output = torch.cat((mix_T1, mix_T1ce,mix_T2,mix_T2f), dim=1)
    return output


def get_cut_mask(out, thres=0.5, nms=0):
    probs = F.softmax(out, 1)
    masks = (probs >= thres).type(torch.int64)
    masks = masks[:, 1, :, :].contiguous()
    if nms == 1:
        masks = LargestCC_pancreas(masks)
    return masks


def LargestCC_pancreas(segmentation):
    N = segmentation.shape[0]
    batch_list = []
    for n in range(N):
        n_prob = segmentation[n].detach().cpu().numpy()
        labels = label(n_prob)
        if labels.max() != 0:
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        else:
            largestCC = n_prob
        batch_list.append(largestCC)

    return torch.Tensor(batch_list).cuda()

def context_mask(img, mask_ratio):
    batch_size, channel, img_x, img_y, img_z = img.shape[0],img.shape[1],img.shape[2],img.shape[3],img.shape[4]
    loss_mask = torch.ones(batch_size, img_x, img_y, img_z).cuda()
    mask = torch.ones(img_x, img_y, img_z).cuda()
    patch_pixel_x, patch_pixel_y, patch_pixel_z = int(img_x*mask_ratio), int(img_y*mask_ratio), int(img_z*mask_ratio)
    w = np.random.randint(0, 112 - patch_pixel_x)
    h = np.random.randint(0, 112 - patch_pixel_y)
    z = np.random.randint(0, 80 - patch_pixel_z)
    mask[w:w+patch_pixel_x, h:h+patch_pixel_y, z:z+patch_pixel_z] = 0
    loss_mask[:, w:w+patch_pixel_x, h:h+patch_pixel_y, z:z+patch_pixel_z] = 0
    return mask.long(), loss_mask.long()
def train(args, snapshot_path):
    base_lr = args.base_lr
    train_data_path = args.root_path
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    num_classes = 4

    def create_model(ema=False):
        # Network definition
        net = net_factory_3d(net_type=args.model, in_chns=1, class_num=num_classes)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model
    if 'multi' in args.mod:
        model = create_model()
        # turb_model = create_model(ema=False)
        # model = net_factory_3d(net_type=args.model, in_chns=1, class_num=num_classes)

        db_train = BraTS2019(base_dir=train_data_path,
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

    # 加载预训练的模型权重
    # pretrained_model_path = '/home/qitam/sdb2/home/qiteam_project/huang/SSL4MIS-master/model/BraTs2019_add_trans/unet_3D_multi/iter_29400_dice_0.8823.pth'
    pretrained_model_path = None
    # model.load_state_dict(torch.load(pretrained_model_path))
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    model.train()
    # turb_model.train()
    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(4)
    dice_loss_1 = DiceLoss_1(4)
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    if pretrained_model_path is not None:
        iter_num = int(os.path.basename(pretrained_model_path).split('_')[1])
    else:
        iter_num = 0
    # max_epoch = max_iterations // len(trainloader) + 1
    max_epoch = (max_iterations-iter_num) // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            if i_batch%2==0:
                volume_batch[:,3,:,:,:]=0
                volume_batch = fill_missing_channel(volume_batch, label_batch)
            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)

            loss_dice = dice_loss(outputs_soft, label_batch.unsqueeze(1))
            loss_ce = ce_loss(outputs, label_batch)
            # with torch.no_grad():
            loss = 0.5 * (loss_dice + loss_ce)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, ' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))
            writer.add_scalar('loss/loss', loss, iter_num)
            if iter_num % 20 == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                image = outputs_soft[0, 1:2, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label',
                                 grid_image, iter_num)

                image = label_batch[0, :, :, 20:61:10].unsqueeze(
                    0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_label',
                                 grid_image, iter_num)

            if iter_num>0 and iter_num % 1 == 0 :
                model.eval()
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
                                  avg_metric[0, 0], iter_num)
                writer.add_scalar('info/val_hd95',
                                  avg_metric[0, 1], iter_num)
                logging.info(
                    'iteration %d : dice_score : %f hd95 : %f' % (iter_num, avg_metric[0, 0].mean(), avg_metric[0, 1].mean()))
                model.train()

            if iter_num % 2000==0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
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
