import math
from glob import glob

# import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from medpy import metric
from tqdm import tqdm
from monai.inferers import SlidingWindowInferer
from monai import transforms

import numpy as np


def cal(o, t):
    # Convert tensors to numpy arrays if they aren't already
    o = np.asarray(o)
    t = np.asarray(t)

    # Compute sums for the conditions
    sum_o = np.sum(o)
    sum_t = np.sum(t)

    # Case 1: True Negative (ground truth = 0 and prediction = 0)
    if sum_t == 0 and sum_o == 0:
        tc = 1.0  # Dice coefficient is perfect when both are zero
        tc_hd = 0.0  # HD95 can be 0 when no positive samples
        # tc_ravd and tc_asd can be set to 0 as they are not applicable
        return np.array([tc, tc_hd])

    # Case 2: False Positive (ground truth = 0 but prediction != 0)
    if sum_t == 0 and sum_o != 0:
        tc = 0.0  # All metrics should be 0
        tc_hd = 0.0
        # tc_ravd and tc_asd can be set to 0
        return np.array([tc, tc_hd])

    # Case 3: False Negative (ground truth != 0 but prediction = 0)
    if sum_t != 0 and sum_o == 0:
        tc = 0.0  # All metrics should be 0
        tc_hd = 0.0
        # tc_ravd and tc_asd can be set to 0
        return np.array([tc, tc_hd])

    # Case 4: Normal Case (compute metrics normally)
    if sum_o != 0 and sum_t != 0:
        tc = dice(o, t)  # Assuming dice is a function that computes the Dice coefficient
        tc_hd = metric.hd95(o, t)  # Assuming metric.hd95 computes the HD95
        # tc_ravd = abs(metric.ravd(o, t))  # Uncomment if you have RAVD metric
        # tc_asd = metric.asd(o, t)  # Uncomment if you have ASD metric
    else:
        tc = 0.0
        tc_hd = 0.0
        # tc_ravd = 0.0
        # tc_asd = 0.0

    return np.array([tc, tc_hd])


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
def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1,mod = None):
    if mod == 'multi_concat':
        c,w, h, d = image.shape
        # if the size of image is less than patch_size, then padding it
        add_pad = False
        if w < patch_size[0]:
            w_pad = patch_size[0]-w
            add_pad = True
        else:
            w_pad = 0
        if h < patch_size[1]:
            h_pad = patch_size[1]-h
            add_pad = True
        else:
            h_pad = 0
        if d < patch_size[2]:
            d_pad = patch_size[2]-d
            add_pad = True
        else:
            d_pad = 0
        wl_pad, wr_pad = w_pad//2, w_pad-w_pad//2
        hl_pad, hr_pad = h_pad//2, h_pad-h_pad//2
        dl_pad, dr_pad = d_pad//2, d_pad-d_pad//2
        if add_pad:
            image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                                   (dl_pad, dr_pad)], mode='constant', constant_values=0)
        # ww, hh, dd = image.shape
        cc,ww, hh, dd = image.shape

        sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
        sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
        sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
        # print("{}, {}, {}".format(sx, sy, sz))
        score_map = np.zeros((num_classes, ) + (image.shape[1],image.shape[2],image.shape[3])).astype(np.float32)
        cnt = np.zeros((image.shape[1],image.shape[2],image.shape[3])).astype(np.float32)
        # label_map_list = []
        for x in range(0, sx):
            xs = min(stride_xy*x, ww-patch_size[0])
            for y in range(0, sy):
                ys = min(stride_xy * y, hh-patch_size[1])
                for z in range(0, sz):
                    zs = min(stride_z * z, dd-patch_size[2])
                    test_patch = image[:,xs:xs+patch_size[0],
                                       ys:ys+patch_size[1], zs:zs+patch_size[2]]
                    test_patch = np.expand_dims(
                        test_patch, axis=0).astype(np.float32)
                    test_patch = torch.from_numpy(test_patch).cuda()
                    with torch.no_grad():
                        y1 = net(test_patch,pred_type="ddim_sample")
                        # ensemble
                        y = torch.softmax(y1, dim=1)
                    y = y.cpu().data.numpy()
                    y = y[0, :, :, :, :]
                    score_map[:,xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                        = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                    cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                        = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
        score_map = score_map/np.expand_dims(cnt, axis=0)
        label_map = np.argmax(score_map, axis=0)
        if add_pad:
            label_map = label_map[wl_pad:wl_pad + w,
                        hl_pad:hl_pad + h, dl_pad:dl_pad + d]
            score_map = score_map[:, wl_pad:wl_pad +
                                            w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        # label_map_list.append(label_map)
        # label_map = np.stack(label_map_list, axis=0)
    return label_map

def test_single_case_multi(net, image, stride_xy, stride_z, patch_size, num_classes=1,mod = None):
    if mod == 'multi_concat':
        c,w, h, d = image.shape
        # if the size of image is less than patch_size, then padding it
        add_pad = False
        if w < patch_size[0]:
            w_pad = patch_size[0]-w
            add_pad = True
        else:
            w_pad = 0
        if h < patch_size[1]:
            h_pad = patch_size[1]-h
            add_pad = True
        else:
            h_pad = 0
        if d < patch_size[2]:
            d_pad = patch_size[2]-d
            add_pad = True
        else:
            d_pad = 0
        wl_pad, wr_pad = w_pad//2, w_pad-w_pad//2
        hl_pad, hr_pad = h_pad//2, h_pad-h_pad//2
        dl_pad, dr_pad = d_pad//2, d_pad-d_pad//2
        if add_pad:
            image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                                   (dl_pad, dr_pad)], mode='constant', constant_values=0)
        cc,ww, hh, dd = image.shape

        sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
        sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
        sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
        score_map = np.zeros((num_classes, ) + (image.shape[1],image.shape[2],image.shape[3])).astype(np.float32)
        cnt = np.zeros((image.shape[1],image.shape[2],image.shape[3])).astype(np.float32)
        label_map_list = []
        for x in range(0, sx):
            xs = min(stride_xy*x, ww-patch_size[0])
            for y in range(0, sy):
                ys = min(stride_xy * y, hh-patch_size[1])
                for z in range(0, sz):
                    zs = min(stride_z * z, dd-patch_size[2])
                    test_patch = image[:,xs:xs+patch_size[0],
                                       ys:ys+patch_size[1], zs:zs+patch_size[2]]
                    test_patch = np.expand_dims(
                        test_patch, axis=0).astype(np.float32)
                    test_patch = torch.from_numpy(test_patch).cuda()
                    encoded_features = []
                    with torch.no_grad():
                        for i in range(4):
                            encoded = net[0][i](test_patch[:, i, :, :, :])
                            encoded_features.append(encoded)
                        concatenated_encoded = [torch.cat([encoded[i] for encoded in encoded_features], dim=1) for i in
                                                range(len(encoded_features[0]))]
                        y1 = net[1](concatenated_encoded[0], concatenated_encoded[1], concatenated_encoded[2], concatenated_encoded[3], concatenated_encoded[4])
                        y = torch.softmax(y1,dim=1)
                        y = y.cpu().data.numpy()
                        y = y[0, :, :, :, :]
                        score_map[:,xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                            = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                        cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                            = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
            score_map = score_map/np.expand_dims(cnt, axis=0)
            label_map = np.argmax(score_map, axis=0)
            if add_pad:
                label_map = label_map[wl_pad:wl_pad + w,
                            hl_pad:hl_pad + h, dl_pad:dl_pad + d]
                score_map = score_map[:, wl_pad:wl_pad +
                                                w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
            label_map_list.append(label_map)
        label_map = np.stack(label_map_list, axis=0)
    return label_map

def cal_metric(gt, pred):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return np.array([dice, hd95])
    else:
        return np.zeros(2)

def calculate_metric_percase(label, label_map):
    def compute_metrics(pred, gt):
        if np.sum(gt) != 0 and np.sum(pred)!=0:
            dice = metric.dc(pred, gt)
            hd = metric.hd95(pred, gt)
        else:
            dice = 0.0
            hd = 0.0

        return np.array([dice, hd])

    # Ground truth binary masks
    et_true = (label == 3).astype(np.float32)
    tc_true = (np.isin(label, [1, 3])).astype(np.float32)
    wt_true = (np.isin(label, [1, 2, 3])).astype(np.float32)

    valid_labels = np.unique(label)

    # Create masks for valid labels in prediction
    et_pred = (label_map == 3).astype(np.float32) if 3 in valid_labels else np.zeros_like(label_map, dtype=np.float32)
    tc_pred = (np.isin(label_map, [1, 3])).astype(np.float32) if any(
        l in valid_labels for l in [1, 3]) else np.zeros_like(label_map, dtype=np.float32)
    wt_pred = (np.isin(label_map, [1, 2, 3])).astype(np.float32) if any(
        l in valid_labels for l in [1, 2, 3]) else np.zeros_like(label_map, dtype=np.float32)

    # Calculate metrics for each class
    et_metrics = compute_metrics(et_pred, et_true)
    tc_metrics = compute_metrics(tc_pred, tc_true)
    wt_metrics = compute_metrics(wt_pred, wt_true)


    return np.array([et_metrics, tc_metrics, wt_metrics])


def test_all_case(net, base_dir, test_list="full_test.list", num_classes=2, patch_size=(48, 160, 160), stride_xy=32, stride_z=24,mod = None):
    window_infer = SlidingWindowInferer(roi_size=[96, 96, 96],
                                             sw_batch_size=1,
                                             overlap=0.5)
    with open(base_dir + '/{}'.format(test_list), 'r') as f:
        image_list = f.readlines()
    if mod == 'multi_concat':
        image_list = [base_dir + "/{}".format(
            item.replace('\n', '').split(",")[0]) for item in image_list]
        total_metric = np.zeros((num_classes, 2))
        print("Validation begin")
        # with open(test_save_path + "/{}.txt".format(method), "a") as f:
        for image_path in tqdm(image_list):
                ids = image_path.split("/")[-1].replace(".h5", "")
                label_path = image_path.replace('t1n', 'seg')
                try:
                    T1 = nib.load(image_path).get_fdata()
                    T1ce = nib.load(image_path.replace(image_path[:-7].split('-')[-1], 't1c')).get_fdata()
                    T2w = nib.load(image_path.replace(image_path[:-7].split('-')[-1], 't2w')).get_fdata()
                    T2f = nib.load(image_path.replace(image_path[:-7].split('-')[-1], 't2f')).get_fdata()
                    image_data = np.array([T1, T1ce, T2w, T2f]).astype(np.float32)

                    label = nib.load(label_path).get_fdata()
                    # seg_data = np.expand_dims(np.array(label).astype(np.int32), axis=0)
                except Exception as e:
                    print(f"加载图像时出错: {e}")
                    continue
                label = [(label == 1) | (label == 3), (label == 1) | (label == 3) | (label == 2), label == 3]
                label = np.stack(label, axis=0).astype(np.float32)
                val = {"image": image_data, "label": label}

                trans_label = val["label"]
                test_patch = np.expand_dims(val["image"],axis = 0)
                test_patch = torch.from_numpy(test_patch).cuda()

                output = window_infer(test_patch, net, pred_type="ddim_sample")
                output = torch.sigmoid(output)
                print('推理完成')

                output = (output > 0.5).float().cpu().numpy()
                target = np.expand_dims(trans_label,axis = 0)
                o = output[:, 1]
                t = target[:, 1]  # ce
                wt = cal(o, t)
                # core
                o = output[:, 0]
                t = target[:, 0]
                tc = cal(o, t)
                # active
                o = output[:, 2]
                t = target[:, 2]
                et = cal(o, t)

                metrics = np.array([wt, tc, et])
                print(f"t is {et},tc is {tc}, ewt is {wt} ")

                # 计算并保存指标
                total_metric += metrics
        print("Validation end")
    return total_metric / len(image_list)

