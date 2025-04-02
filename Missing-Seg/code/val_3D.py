import math
from glob import glob

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from medpy import metric
from tqdm import tqdm


def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1,mod = None):
    if 'multi' not in mod:
        w, h, d = image.shape
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
        ww, hh, dd = image.shape

        sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
        sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
        sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
        # print("{}, {}, {}".format(sx, sy, sz))
        score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
        cnt = np.zeros(image.shape).astype(np.float32)

        for x in range(0, sx):
            xs = min(stride_xy*x, ww-patch_size[0])
            for y in range(0, sy):
                ys = min(stride_xy * y, hh-patch_size[1])
                for z in range(0, sz):
                    zs = min(stride_z * z, dd-patch_size[2])
                    test_patch = image[xs:xs+patch_size[0],
                                       ys:ys+patch_size[1], zs:zs+patch_size[2]]
                    test_patch = np.expand_dims(np.expand_dims(
                        test_patch, axis=0), axis=0).astype(np.float32)
                    test_patch = torch.from_numpy(test_patch).cuda()

                    with torch.no_grad():
                        y1 = net(test_patch)
                        # ensemble
                        y = torch.softmax(y1, dim=1)
                    y = y.cpu().data.numpy()
                    y = y[0, :, :, :, :]
                    score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                        = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                    cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                        = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
        score_map = score_map/np.expand_dims(cnt, axis=0)
        label_map = np.argmax(score_map, axis=0)

        if add_pad:
            label_map = label_map[wl_pad:wl_pad+w,
                                  hl_pad:hl_pad+h, dl_pad:dl_pad+d]
            score_map = score_map[:, wl_pad:wl_pad +
                                  w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        # return label_map
    elif mod == 'multi_concat':
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
        label_map_list = []
        # for i in range(4):
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
                            # encoded = net[0][i](test_patch[:, i, :, :, :])
                            # y1 = net[1](encoded[0], encoded[1], encoded[2], encoded[3], encoded[4])
                            # y = torch.softmax(y1,dim=1)
                            y1 = net(test_patch)
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
def calculate_metric_percase(label, label_map):
    def compute_metrics(pred, gt):
        # gt 和 pred 都为 0 的情况
        if np.sum(gt) == 0 and np.sum(pred) == 0:
            dice = 1.0
            hd = 1.0  # 或者对于 hd，可以设为任意较大值，因为无实际意义
        # gt 为 0，pred 不为 0 的情况，跳过计算
        elif np.sum(gt) == 0 and np.sum(pred) != 0:
            return None  # 或者返回其他特殊标志，例如 NaN 或 "skip"
        # gt 不为 0，pred 为 0 的情况，计算为 0
        elif np.sum(gt) != 0 and np.sum(pred) == 0:
            dice = 0.0
            hd = 0.0
        # 正常情况，gt 和 pred 都不为 0 时，计算指标
        else:
            dice = metric.dc(pred, gt)
            hd = metric.hd95(pred, gt)

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


def test_all_case(net, base_dir, test_list="full_test.list", num_classes=4, patch_size=(48, 160, 160), stride_xy=32, stride_z=24,mod = None):
    with open(base_dir + '/{}'.format(test_list), 'r') as f:
        image_list = f.readlines()
    if 'multi' not in mod:
        image_list = [base_dir + "/{}/{}".format(
            mod.upper(), item.replace('\n', '').split(",")[0]) for item in image_list]
    # image_list = [base_dir + "/data/{}.h5".format(
    #     item.replace('\n', '').split(",")[0]) for item in image_list]
        total_metric = np.zeros((num_classes-1, 2))
        print("Validation begin")
        for image_path in tqdm(image_list):
            # h5f = h5py.File(image_path, 'r')
            # image = h5f['image'][:]
            # label = h5f['label'][:]
            label_path = image_path.replace(mod.upper(), 'mask').replace(image_path[-10:-7],'seg')
            # label_path = image_path.replace(mod,'seg')
            image = nib.load(image_path).get_fdata()
            label = nib.load(label_path).get_fdata()
            # 处理成二分类
            # label[label != 0] = 1
            # h5f = h5py.File(image_path, 'r')
            # image = h5f['image'][:]
            # label = h5f['label'][:]
            prediction = test_single_case(
                net, image, stride_xy, stride_z, patch_size, num_classes=num_classes,mod = mod)
            for i in range(1, num_classes):
                total_metric[i-1, :] += cal_metric(label == i, prediction == i)
        print("Validation end")
    elif mod == 'multi':
        image_list = [base_dir + "/{}".format(
            item.replace('\n', '').split(",")[0]) for item in image_list]
        total_metric = np.zeros((num_classes - 1, 2))
        print("Validation begin")
        for image_path in tqdm(image_list):
            # h5f = h5py.File(image_path, 'r')
            # image = h5f['image'][:]
            # label = h5f['label'][:]
            label_path = image_path.replace('t1n', 'seg')
            T1 = nib.load(image_path).get_fdata()
            T1ce = nib.load(image_path.replace(image_path[:-7].split('-')[-1], 't1c')).get_fdata()
            T2w = nib.load(image_path.replace(image_path[:-7].split('-')[-1], 't2w')).get_fdata()
            T2f = nib.load(image_path.replace(image_path[:-7].split('-')[-1], 't2f')).get_fdata()
            image = 0.25*T1+0.25*T1ce+0.25*T2w+0.25*T2f
            # image = nib.load(image_path).get_fdata()
            label = nib.load(label_path).get_fdata()
            # 处理成二分类
            label[label != 0] = 1
            # h5f = h5py.File(image_path, 'r')
            # image = h5f['image'][:]
            # label = h5f['label'][:]
            prediction = test_single_case(
                net, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
            for i in range(1, num_classes):
                total_metric[i - 1, :] += cal_metric(label == i, prediction == i)
        print("Validation end")
    elif mod == 'multi_concat':
        mean_metrics = np.zeros((num_classes - 1, 2))
        image_list = [base_dir + "/{}".format(
            item.replace('\n', '').split(",")[0]) for item in image_list]
        total_metric = np.zeros((num_classes - 1, 2))
        print("Validation begin")
        valid_image_count = 0
        for image_path in tqdm(image_list):
            label_path = image_path.replace('t1n', 'seg')
            T1 = nib.load(image_path).get_fdata()
            T1ce = nib.load(image_path.replace(image_path[:-7].split('-')[-1], 't1c')).get_fdata()
            T2w = nib.load(image_path.replace(image_path[:-7].split('-')[-1], 't2w')).get_fdata()
            T2f = nib.load(image_path.replace(image_path[:-7].split('-')[-1], 't2f')).get_fdata()
            # image = 0.25*T1+0.25*T1ce+0.25*T2w+0.25*T2f
            image = np.stack((T1,T1ce,T2w,T2f),axis=0)
            # image = nib.load(image_path).get_fdata()
            label = nib.load(label_path).get_fdata()
            # 处理成二分类
            # label[label != 0] = 1
            # h5f = h5py.File(image_path, 'r')
            # image = h5f['image'][:]
            # label = h5f['label'][:]
            prediction = test_single_case(
                net, image, stride_xy, stride_z, patch_size, num_classes=num_classes,mod=mod)
            metrics = calculate_metric_percase(label, prediction)
            if metrics is not None and not np.isnan(metrics).any():  # 如果 metrics 不包含 None 或 NaN
                total_metric += metrics  # 累加指标
                valid_image_count += 1  #
            else:
                print('存在缺失值')
        if valid_image_count > 0:
            mean_metrics = total_metric / valid_image_count
            # for i in range(1, num_classes):
            #     # print(prediction.max())
            #     for j in range(prediction.shape[0]):# j是第j个label
            #         total_metric[i - 1, :] += cal_metric(label == i, prediction[j,:,:,:] == i)
            #         # a = cal_metric(label == i, prediction[j,:,:,:] == i)
            #     # total_metric = total_metric/4
        print("Validation end")
    # return total_metric / (len(image_list)*4)
    return mean_metrics

def test_all_case_multi(net, base_dir, test_list="full_test.list", num_classes=2, patch_size=(48, 160, 160), stride_xy=32, stride_z=24,mod = None):
    with open(base_dir + '/{}'.format(test_list), 'r') as f:
        image_list = f.readlines()
    if mod == 'multi_concat':
        image_list = [base_dir + "/{}".format(
            item.replace('\n', '').split(",")[0]) for item in image_list]
        total_metric = np.zeros((num_classes - 1, 2))
        print("Validation begin")
        for image_path in tqdm(image_list):
            label_path = image_path.replace('t1n', 'seg')
            T1 = nib.load(image_path).get_fdata()
            T1ce = nib.load(image_path.replace(image_path[:-7].split('-')[-1], 't1c')).get_fdata()
            T2w = nib.load(image_path.replace(image_path[:-7].split('-')[-1], 't2w')).get_fdata()
            T2f = nib.load(image_path.replace(image_path[:-7].split('-')[-1], 't2f')).get_fdata()
            image = np.stack((T1,T1ce,T2w,T2f),axis=0)
            label = nib.load(label_path).get_fdata()
            # 处理成二分类
            label[label != 0] = 1
            prediction = test_single_case_multi(
                net, image, stride_xy, stride_z, patch_size, num_classes=num_classes,mod=mod)
            for i in range(1, num_classes):
                for j in range(prediction.shape[0]):# j是第j个label
                    total_metric[i - 1, :] += cal_metric(label == i, prediction[j,:,:,:] == i)
        print("Validation end")
    return total_metric / len(image_list)
