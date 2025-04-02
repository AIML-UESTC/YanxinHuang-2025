import argparse
import logging
import os
import random
import shutil
import sys
import time
# from skimage.measure import label
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
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
# from dataloaders.brats_37 import (BraTS2019, CenterCrop, RandomCrop,
#                                    RandomRotFlip, ToTensor,RandomRotFlip_multi,RandomCrop_multi,ToTensor_multi,
#                                    TwoStreamBatchSampler)
from networks.net_factory_3d import net_factory_3d
from utils import losses, metrics, ramps
# from val_3D import test_all_case
from val_diff import test_all_case
import os
import cp_losses
from utils import losses
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/root/autodl-tmp/BraTS2023-GLITraining', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='2023_diff', help='experiment_name')
parser.add_argument('--mod', type=str,
                    default='multi_concat', help='experiment_mod')
parser.add_argument('--model', type=str,
                    default='diff', help='model_name')
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
def train(args, snapshot_path):
    base_lr = args.base_lr
    train_data_path = args.root_path
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    num_classes = 3

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
                             ]),mod = args.mod)
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    # 加载预训练的模型权重
    # pretrained_model_path = '/root/SSL4MIS-master/model/2023_diff_more/diff/iter_40000.pth'
    pretrained_model_path = None
    # model.load_state_dict(torch.load(pretrained_model_path))
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=16, pin_memory=False, worker_init_fn=worker_init_fn)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ai_dice = mon_Diceloss(sigmoid=True,include_background=False)

    mse = nn.MSELoss()
    """
    这里注意区别nn.BCEWithLogitsLoss()与nn.CrossEntropyLoss()的区别。
    1.nn.BCEWithLogitsLoss()支持多标签的损失计算，nn.CrossEntropyLoss()不支持
    2.nn.BCEWithLogitsLoss()内置的激活函数是Sigmoid函数，nn.CrossEntropyLoss()内置激活函数为Softmax()
    在这个方法中针对多模态多标签的分割任务，选用BCE是合理的选择。在调试时也会发现由于预处理数据时涉及Crop操作降低数据量，选用[96,96,96]来裁剪数据时会出现无标签出现的情形。
    """
    bce = nn.BCEWithLogitsLoss()
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    if pretrained_model_path is not None:
        iter_num = int(os.path.basename(pretrained_model_path).split('_')[1][:-4])
    else:
        iter_num = 0
    # max_epoch = max_iterations // len(trainloader) + 1
    total_iters = args.max_iterations
    half_iters = total_iters // 2
    base_interval = 1000
    min_interval = 500
    patience = 5  # 早停耐心值
    no_improve_count = 0
    max_epoch = (max_iterations-iter_num) // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            image = volume_batch
            x_start=label_batch.float()
            x_start = (x_start) * 2 - 1
            x_t, t, noise = model(x=x_start, pred_type="q_sample")
            pred_xstart = model(x=x_t, step=t, image=image, pred_type="denoise")
            # outputs_soft = torch.softmax(pred_xstart, dim=1)

            loss_bce = bce(pred_xstart, label_batch)
            loss_dice = ai_dice(pred_xstart, label_batch)

            pred_xstart = torch.sigmoid(pred_xstart)
            loss_mse = mse(pred_xstart, label_batch)

            loss = loss_dice + loss_bce + loss_mse

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/loss_bce', loss_bce, iter_num)
            writer.add_scalar('info/loss_mse', loss_mse, iter_num)

            logging.info('iteration %d : loss : %f, loss_dice: %f, loss_bce: %f,loss_mse:%f' %
                (iter_num, loss.item(), loss_dice.item(),loss_bce.item(),loss_mse.item()))
            writer.add_scalar('loss/loss', loss, iter_num)
            if iter_num % 2000 == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/T1Image', grid_image, iter_num)

                label_tc = pred_xstart[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(label_tc, 5, normalize=False)
                writer.add_image('train/Predicted_label_tc',
                                 grid_image, iter_num)
                label_wt = pred_xstart[0, 1:2, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(label_wt, 5, normalize=False)
                writer.add_image('train/Predicted_label_wt',
                                 grid_image, iter_num)
                label_et = pred_xstart[0, 2:3, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(label_et, 5, normalize=False)
                writer.add_image('train/Predicted_label_et',
                                 grid_image, iter_num)
            """
            验证阶段，是重要的阶段，和测试代码有许多重合。
            """
            # if iter_num > 10000:  # 初始阶段跳过验证
                # 动态计算验证间隔（线性衰减）
                # if iter_num < half_iters:
                #     val_interval = base_interval
                # else:
                #     progress = (iter_num - half_iters) / (total_iters - half_iters)
                    # val_interval = max(min_interval, int(base_interval * (1 - 0.75 * progress)))

                # 执行验证
                # if iter_num % 200 == 0:
                # # if iter_num % 1 == 0:
                #     model.eval()
                #     avg_metric = test_all_case(model, args.root_path, test_list="val.txt",
                #                                num_classes=num_classes, patch_size=args.patch_size,
                #                                stride_xy=64, stride_z=64, mod=args.mod)
                #
                #     current_dice = avg_metric[:, 0].mean()
                #     # print(f'Iter {iter_num}: Dice={current_dice:.4f} (Interval={val_interval})')
                #     print(f'Iter {iter_num}: Dice={current_dice:.4f} ')
                #
                #     # 模型保存逻辑
                #     if current_dice > best_performance:
                #         best_performance = current_dice
                #         no_improve_count = 0  # 重置计数器
                #
                #         torch.save(model.state_dict(),
                #                    os.path.join(snapshot_path, f'iter_{iter_num}_dice_{current_dice:.4f}.pth'))
                #         torch.save(model.state_dict(),
                #                    os.path.join(snapshot_path, f'{args.model}_best_model.pth'))
                #     else:
                #         no_improve_count += 1

                    # 早停判断
                    # if no_improve_count >= patience:
                    #     print(f'Early stopping at iter {iter_num} (no improvement for {patience} validations)')
                    #     break
                    # writer.add_scalar('info/val_dice_score',
                    #                   avg_metric[:,0].mean(), iter_num)
                    # writer.add_scalar('info/val_hd95',
                    #                  avg_metric[:,1].mean(), iter_num)
                    # logging.info(
                    #     'iteration %d : dice_score : %f' % (iter_num, avg_metric[:,0].mean()))
                    # model.train()

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
