import math

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from medpy import metric
from skimage.measure import label
from tqdm import tqdm


def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1,mod=None):
    if 'multi' not in mod and 'fuse' not in mod:
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
    elif mod =='multi_concat':
        c, w, h, d = image.shape

        # if the size of image is less than patch_size, then padding it
        add_pad = False
        if w < patch_size[0]:
            w_pad = patch_size[0] - w
            add_pad = True
        else:
            w_pad = 0
        if h < patch_size[1]:
            h_pad = patch_size[1] - h
            add_pad = True
        else:
            h_pad = 0
        if d < patch_size[2]:
            d_pad = patch_size[2] - d
            add_pad = True
        else:
            d_pad = 0
        wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
        hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
        dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
        if add_pad:
            image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                                   (dl_pad, dr_pad)], mode='constant', constant_values=0)
        cc, ww, hh, dd = image.shape

        sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
        sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
        sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
        # print("{}, {}, {}".format(sx, sy, sz))
        # score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
        score_map = np.zeros((num_classes,) + (image.shape[1], image.shape[2], image.shape[3])).astype(np.float32)
        # cnt = np.zeros(image.shape).astype(np.float32)
        cnt = np.zeros((image.shape[1], image.shape[2], image.shape[3])).astype(np.float32)

        for x in range(0, sx):
            xs = min(stride_xy * x, ww - patch_size[0])
            for y in range(0, sy):
                ys = min(stride_xy * y, hh - patch_size[1])
                for z in range(0, sz):
                    zs = min(stride_z * z, dd - patch_size[2])
                    test_patch = image[:, xs:xs + patch_size[0],
                                 ys:ys + patch_size[1], zs:zs + patch_size[2]]
                    test_patch = np.expand_dims(
                        test_patch, axis=0).astype(np.float32)
                    test_patch = torch.from_numpy(test_patch).cuda()

                    with torch.no_grad():
                        y1 = net(test_patch)
                        # ensemble
                        y = torch.softmax(y1, dim=1)
                    y = y.cpu().data.numpy()
                    y = y[0, :, :, :, :]
                    score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                        = score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + y
                    cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                        = cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + 1
        score_map = score_map / np.expand_dims(cnt, axis=0)
        label_map = np.argmax(score_map, axis=0)

        if add_pad:
            label_map = label_map[wl_pad:wl_pad + w,
                        hl_pad:hl_pad + h, dl_pad:dl_pad + d]
            score_map = score_map[:, wl_pad:wl_pad +
                                            w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
    elif mod =='fuse':
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
        ww, hh, dd = image.shape

        sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
        sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
        sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
        # print("{}, {}, {}".format(sx, sy, sz))
        # score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
        score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
        # cnt = np.zeros(image.shape).astype(np.float32)
        cnt = np.zeros(image.shape).astype(np.float32)

        for x in range(0, sx):
            xs = min(stride_xy*x, ww-patch_size[0])
            for y in range(0, sy):
                ys = min(stride_xy * y, hh-patch_size[1])
                for z in range(0, sz):
                    zs = min(stride_z * z, dd-patch_size[2])
                    test_patch = image[xs:xs+patch_size[0],
                                       ys:ys+patch_size[1], zs:zs+patch_size[2]]
                    test_patch = np.expand_dims(np.expand_dims(test_patch, axis=0)
                    ,axis = 0).astype(np.float32)
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
    return label_map

def test_single_case_multi_out(net, image, stride_xy, stride_z, patch_size, num_classes=1,mod=None):
    if mod =='multi_concat':
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
        sub_map_list = []
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
                    feature = {}
                    conv4 = {}
                    conv3 = {}
                    conv2 = {}
                    conv1 = {}
                    with torch.no_grad():
                        for i in range(4):
                            encoded = net[0][i](test_patch[:, i, :, :, :])
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
                        concat_feature = torch.cat((feature['T1'], feature['T1ce'], feature['T2'], feature['T2f']),
                                                   dim=1)
                        concat_conv4 = torch.cat((conv4['T1'], conv4['T1ce'], conv4['T2'], conv4['T2f']), dim=1)
                        concat_conv3 = torch.cat((conv3['T1'], conv3['T1ce'], conv3['T2'], conv3['T2f']), dim=1)
                        concat_conv2 = torch.cat((conv2['T1'], conv2['T1ce'], conv2['T2'], conv2['T2f']), dim=1)
                        concat_conv1 = torch.cat((conv1['T1'], conv1['T1ce'], conv1['T2'], conv1['T2f']), dim=1)
                        y1 = net[1](concat_feature, concat_conv4, concat_conv3, concat_conv2, concat_conv1)
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
    return label_map


def cal_metric(gt, pred):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return np.array([dice, hd95])
    else:
        return np.zeros(2)


def test_all_case(net, base_dir, method="unet_3D", test_list="full_test.list", num_classes=4, patch_size=(48, 160, 160), stride_xy=32, stride_z=24, test_save_path=None,mod = None):
    with open(base_dir + '/{}'.format(test_list), 'r') as f:
        image_list = f.readlines()
    # image_list = [base_dir + "/data/{}.h5".format(
    #     item.replace('\n', '').split(",")[0]) for item in image_list]
    if  'multi' not in mod and 'fuse' not in mod:
        image_list = [base_dir + "/{}/{}".format(
           mod.upper() ,item.replace('\n', '').split(",")[0]) for item in image_list]
        total_metric = np.zeros((num_classes-1, 4))
        print("Testing begin")
        with open(test_save_path + "/{}.txt".format(method), "a") as f:
            for image_path in tqdm(image_list):
                ids = image_path.split("/")[-1].replace(".h5", "")
                # h5f = h5py.File(image_path, 'r')
                # image = h5f['image'][:]
                # label = h5f['label'][:]
                label_path = image_path.replace(mod.upper(), '/mask').replace(mod, 'seg')
                image = nib.load(image_path).get_fdata()
                label = nib.load(label_path).get_fdata()
                # 处理成二分类
                label[label != 0] = 1
                prediction = test_single_case(
                    net, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
                if np.any(prediction):
                    metric = calculate_metric_percase(prediction == 1, label == 1)
                total_metric[0, :] += metric
                f.writelines("{},{},{},{},{}\n".format(
                    ids, metric[0], metric[1], metric[2], metric[3]))

                pred_itk = sitk.GetImageFromArray(prediction.astype(np.uint8))
                pred_itk.SetSpacing((1.0, 1.0, 1.0))
                sitk.WriteImage(pred_itk, test_save_path +
                                "/{}_pred.nii.gz".format(ids))

                img_itk = sitk.GetImageFromArray(image)
                img_itk.SetSpacing((1.0, 1.0, 1.0))
                sitk.WriteImage(img_itk, test_save_path +
                                "/{}_img.nii.gz".format(ids))

                lab_itk = sitk.GetImageFromArray(label.astype(np.uint8))
                lab_itk.SetSpacing((1.0, 1.0, 1.0))
                sitk.WriteImage(lab_itk, test_save_path +
                                "/{}_lab.nii.gz".format(ids))
            f.writelines("Mean metrics,{},{},{},{}".format(total_metric[0, 0] / len(image_list), total_metric[0, 1] / len(
                image_list), total_metric[0, 2] / len(image_list), total_metric[0, 3] / len(image_list)))
        f.close()
        print("Testing end")
    elif mod == 'fuse':
        image_list = [base_dir + "/{}/{}".format(
           mod.upper() ,item.replace('\n', '').split(",")[0]) for item in image_list]
        total_metric = np.zeros((num_classes-1, 4))
        print("Testing begin")
        with open(test_save_path + "/{}.txt".format(method), "a") as f:
            for image_path in tqdm(image_list):
                ids = image_path.split("/")[-1].replace(".h5", "")
                # h5f = h5py.File(image_path, 'r')
                # image = h5f['image'][:]
                # label = h5f['label'][:]
                label_path = image_path.replace(mod.upper(), 'mask').replace(image_path[-10:-7],'seg')
                image = nib.load(image_path).get_fdata()
                label = nib.load(label_path).get_fdata()
                # 处理成二分类
                label[label != 0] = 1
                prediction = test_single_case(
                    net, image, stride_xy, stride_z, patch_size, num_classes=num_classes,mod = mod)
                if np.any(prediction):
                    metric = calculate_metric_percase(prediction == 1, label == 1)
                total_metric[0, :] += metric
                f.writelines("{},{},{},{},{}\n".format(
                    ids, metric[0], metric[1], metric[2], metric[3]))

                pred_itk = sitk.GetImageFromArray(prediction.astype(np.uint8))
                pred_itk.SetSpacing((1.0, 1.0, 1.0))
                sitk.WriteImage(pred_itk, test_save_path +
                                "/{}_pred.nii.gz".format(ids))

                img_itk = sitk.GetImageFromArray(image)
                img_itk.SetSpacing((1.0, 1.0, 1.0))
                sitk.WriteImage(img_itk, test_save_path +
                                "/{}_img.nii.gz".format(ids))

                lab_itk = sitk.GetImageFromArray(label.astype(np.uint8))
                lab_itk.SetSpacing((1.0, 1.0, 1.0))
                sitk.WriteImage(lab_itk, test_save_path +
                                "/{}_lab.nii.gz".format(ids))
            f.writelines("Mean metrics,{},{},{},{}".format(total_metric[0, 0] / len(image_list), total_metric[0, 1] / len(
                image_list), total_metric[0, 2] / len(image_list), total_metric[0, 3] / len(image_list)))
        f.close()
        print("Testing end")

    elif mod == 'multi':
        image_list = [base_dir + "/{}".format(
        item.replace('\n', '').split(",")[0]) for item in image_list]
        total_metric = np.zeros((num_classes-1, 4))
        print("Testing begin")
        with open(test_save_path + "/{}.txt".format(method), "a") as f:
            for image_path in tqdm(image_list):
                ids = image_path.split("/")[-1].replace(".h5", "")
                # h5f = h5py.File(image_path, 'r')
                # image = h5f['image'][:]
                # label = h5f['label'][:]
                label_path = image_path.replace('t1n', 'seg')

                # image = nib.load(image_path).get_fdata()
                # label = nib.load(label_path).get_fdata()
                T1 = nib.load(image_path).get_fdata()
                T1ce = nib.load(image_path.replace(image_path[:-7].split('-')[-1], 't1c')).get_fdata()
                T2w = nib.load(image_path.replace(image_path[:-7].split('-')[-1], 't2w')).get_fdata()
                T2f = nib.load(image_path.replace(image_path[:-7].split('-')[-1], 't2f')).get_fdata()
                image = 0.25 * T1 + 0.25 * T1ce + 0.25 * T2w + 0.25 * T2f
                # image = nib.load(image_path).get_fdata()
                label = nib.load(label_path).get_fdata()
                # 处理成二分类
                label[label != 0] = 1
                prediction = test_single_case(
                    net, image, stride_xy, stride_z, patch_size, num_classes=num_classes,mod = mod)
                if np.any(prediction):
                    metric = calculate_metric_percase(prediction == 1, label == 1)
                total_metric[0, :] += metric
                f.writelines("{},{},{},{},{}\n".format(
                    ids, metric[0], metric[1], metric[2], metric[3]))

                pred_itk = sitk.GetImageFromArray(prediction.astype(np.uint8))
                pred_itk.SetSpacing((1.0, 1.0, 1.0))
                sitk.WriteImage(pred_itk, test_save_path +
                                "/{}_pred.nii.gz".format(ids))

                img_itk = sitk.GetImageFromArray(image)
                img_itk.SetSpacing((1.0, 1.0, 1.0))
                sitk.WriteImage(img_itk, test_save_path +
                                "/{}_img.nii.gz".format(ids))

                lab_itk = sitk.GetImageFromArray(label.astype(np.uint8))
                lab_itk.SetSpacing((1.0, 1.0, 1.0))
                sitk.WriteImage(lab_itk, test_save_path +
                                "/{}_lab.nii.gz".format(ids))
            f.writelines("Mean metrics,{},{},{},{}".format(total_metric[0, 0] / len(image_list), total_metric[0, 1] / len(
                image_list), total_metric[0, 2] / len(image_list), total_metric[0, 3] / len(image_list)))
        f.close()
    elif mod == 'multi_concat':
        image_list = [base_dir + "/{}".format(
        item.replace('\n', '').split(",")[0]) for item in image_list]
        total_metric = np.zeros((num_classes-1, 4))
        print("Testing begin")
        with open(test_save_path + "/{}.txt".format(method), "a") as f:
            for image_path in tqdm(image_list):
                ids = image_path.split("/")[-1].replace(".h5", "")
                # h5f = h5py.File(image_path, 'r')
                # image = h5f['image'][:]
                # label = h5f['label'][:]
                label_path = image_path.replace('t1n', 'seg')

                # image = nib.load(image_path).get_fdata()
                # label = nib.load(label_path).get_fdata()
                T1 = nib.load(image_path).get_fdata()
                T1ce = nib.load(image_path.replace(image_path[:-7].split('-')[-1], 't1c')).get_fdata()
                T2w = nib.load(image_path.replace(image_path[:-7].split('-')[-1], 't2w')).get_fdata()

                T2f = nib.load(image_path.replace(image_path[:-7].split('-')[-1], 't2f')).get_fdata()
                # T2f = np.zeros_like(T1)
                # T2f = (T1+T1ce+T2w)/3
                # image = 0.25 * T1 + 0.25 * T1ce + 0.25 * T2w + 0.25 * T2f
                label = nib.load(label_path).get_fdata()
                # nonzero_indices = np.nonzero(label)

                # 使用这些位置从 T2w 提取相应的部分
                # T2w_patch = T2w[nonzero_indices]

                # 复制 T2w_patch 到 T1ce 的相应位置，生成新的 T2f
                # T2f = T1ce.copy()
                # T2f[nonzero_indices] = T2w_patch
                # T2f =
                image = np.stack((T1, T1ce, T2w, T2f), axis=0)
                # image = nib.load(image_path).get_fdata()

                # 处理成二分类
                # label[label != 0] = 1
                prediction = test_single_case(
                    net, image, stride_xy, stride_z, patch_size, num_classes=num_classes,mod=mod)
                if np.any(prediction):
                    if np.any(prediction == 1) and np.any(label == 1):
                        metric = calculate_metric_percase(prediction==1, label==1)
                        total_metric[0, :] += metric
                    if np.any(prediction == 2) and np.any(label == 2):
                        metric_2 = calculate_metric_percase(prediction==2, label==2)
                        total_metric[1, :] += metric_2
                    if np.any(prediction == 3) and np.any(label == 3):
                        metric_3 = calculate_metric_percase(prediction==3, label==3)
                        total_metric[2, :] += metric_3
                # total_metric[0, :] += metric
                # total_metric[1, :] += metric_2
                # total_metric[2, :] += metric_3
                f.writelines("{},{},{},{},{}\n".format(
                    ids, metric[0], metric[1], metric[2], metric[3]))

                pred_itk = sitk.GetImageFromArray(prediction.astype(np.uint8))
                pred_itk.SetSpacing((1.0, 1.0, 1.0))
                sitk.WriteImage(pred_itk, test_save_path +
                                "/{}_pred.nii.gz".format(ids))

                img_itk = sitk.GetImageFromArray(image)
                img_itk.SetSpacing((1.0, 1.0, 1.0))
                sitk.WriteImage(img_itk, test_save_path +
                                "/{}_img.nii.gz".format(ids))

                lab_itk = sitk.GetImageFromArray(label.astype(np.uint8))
                lab_itk.SetSpacing((1.0, 1.0, 1.0))
                sitk.WriteImage(lab_itk, test_save_path +
                                "/{}_lab.nii.gz".format(ids))
            f.writelines("Mean metrics,{},{},{},{}".format(total_metric[0, 0] / len(image_list), total_metric[0, 1] / len(
                image_list), total_metric[0, 2] / len(image_list), total_metric[0, 3] / len(image_list)))
        f.close()
        print("Testing end")

    return total_metric / len(image_list)

def test_all_case_multi(net, base_dir, method="unet_3D", test_list="full_test.list", num_classes=4, patch_size=(48, 160, 160), stride_xy=32, stride_z=24, test_save_path=None,mod = None):
    with open(base_dir + '/{}'.format(test_list), 'r') as f:
        image_list = f.readlines()
    if mod == 'multi_concat':
        image_list = [base_dir + "/{}".format(
        item.replace('\n', '').split(",")[0]) for item in image_list]
        total_metric = np.zeros((num_classes-1, 4))
        print("Testing begin")
        with open(test_save_path + "/{}.txt".format(method), "a") as f:
            for image_path in tqdm(image_list):
                ids = image_path.split("/")[-1].replace(".h5", "")
                label_path = image_path.replace('t1n', 'seg')
                T1 = nib.load(image_path).get_fdata()
                T1ce = nib.load(image_path.replace(image_path[:-7].split('-')[-1], 't1c')).get_fdata()
                T2w = nib.load(image_path.replace(image_path[:-7].split('-')[-1], 't2w')).get_fdata()
                T2f = nib.load(image_path.replace(image_path[:-7].split('-')[-1], 't2f')).get_fdata()
                image = np.stack((T1, T1ce, T2w, T2f), axis=0)
                label = nib.load(label_path).get_fdata()
                # 处理成二分类
                label[label != 0] = 1
                prediction = test_single_case_multi_out(
                    net, image, stride_xy, stride_z, patch_size, num_classes=num_classes,mod=mod)
                # x = prediction
                # ids_list = [ids,ids.replace('t1n','t1c'),ids.replace('t1n','t2w'),ids.replace('t1n','t2f')]
                # ids = ids_list[i]
                # prediction = x[i]
                if np.any(prediction):
                    metric = calculate_metric_percase(prediction == 1, label == 1)
                total_metric[0, :] += metric
                f.writelines("{},{},{},{},{}\n".format(
                    ids, metric[0], metric[1], metric[2], metric[3]))

                pred_itk = sitk.GetImageFromArray(prediction.astype(np.uint8))
                pred_itk.SetSpacing((1.0, 1.0, 1.0))
                sitk.WriteImage(pred_itk, test_save_path +
                                "/{}_pred.nii.gz".format(ids))

                img_itk = sitk.GetImageFromArray(image)
                img_itk.SetSpacing((1.0, 1.0, 1.0))
                sitk.WriteImage(img_itk, test_save_path +
                                "/{}_img.nii.gz".format(ids))

                lab_itk = sitk.GetImageFromArray(label.astype(np.uint8))
                lab_itk.SetSpacing((1.0, 1.0, 1.0))
                sitk.WriteImage(lab_itk, test_save_path +
                                "/{}_lab.nii.gz".format(ids))
            f.writelines("Mean metrics,{},{},{},{}".format(total_metric[0, 0] / len(image_list), total_metric[0, 1] / len(
                image_list), total_metric[0, 2] / len(image_list), total_metric[0, 3] / len(image_list) ) )
        f.close()
        print("Testing end")

    return total_metric / len(image_list)


def cal_dice(prediction, label, num=2):
    total_dice = np.zeros(num-1)
    for i in range(1, num):
        prediction_tmp = (prediction == i)
        label_tmp = (label == i)
        prediction_tmp = prediction_tmp.astype(np.float)
        label_tmp = label_tmp.astype(np.float)

        dice = 2 * np.sum(prediction_tmp * label_tmp) / \
            (np.sum(prediction_tmp) + np.sum(label_tmp))
        total_dice[i - 1] += dice

    return total_dice


def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    ravd = abs(metric.binary.ravd(pred, gt))
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)
    return np.array([dice, ravd, hd, asd])
