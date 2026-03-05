from __future__ import print_function

import argparse
import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import numpy as np
from PIL import Image

from models.model_zoo import get_segmentation_model
from utils.score import SegmentationMetric
from utils.visualize import get_color_pallete
from dataset.voc import VOCDataValSet


def parse_args():
    parser = argparse.ArgumentParser(description='VOC fixed-set test with mIoU + result saving')
    parser.add_argument('--model', type=str, default='deeplabv3', help='model name')
    parser.add_argument('--backbone', type=str, default='resnet18', help='backbone name')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to trained model state_dict (.pth/.pkl)')

    parser.add_argument('--data', type=str, default='./dataset/VOCAug/',
                        help='VOC root directory (contains JPEGImages/ and SegmentationClass/)')
    parser.add_argument('--data-list', type=str, default='./dataset/list/voc/val.txt',
                        help='fixed evaluation list file')
    parser.add_argument('--workers', '-j', type=int, default=8, metavar='N', help='dataloader threads')

    parser.add_argument('--save-dir', type=str, default='./runs/voc_test_results',
                        help='directory to save outputs')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disable CUDA')
    return parser.parse_args()


def build_model(args, num_class, device):
    aux = args.backbone.startswith('resnet')
    batchnorm_layer = nn.BatchNorm2d

    if 'former' in args.model:
        model = get_segmentation_model(
            model=args.model,
            backbone=args.backbone,
            pretrained='None',
            batchnorm_layer=batchnorm_layer,
            num_class=num_class,
        ).to(device)
    else:
        model = get_segmentation_model(
            model=args.model,
            backbone=args.backbone,
            aux=aux,
            pretrained='None',
            pretrained_base='None',
            local_rank=0,
            norm_layer=batchnorm_layer,
            num_class=num_class,
        ).to(device)

    state = torch.load(args.checkpoint, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']

    cleaned_state = {}
    for key, value in state.items():
        if key.startswith('module.'):
            cleaned_state[key[7:]] = value
        else:
            cleaned_state[key] = value

    model.load_state_dict(cleaned_state, strict=True)

    model.eval()
    return model


def save_outputs(image_path, pred_np, gt_np, name, out_root):
    image_dir = os.path.join(out_root, 'image')
    pred_dir = os.path.join(out_root, 'pred')
    gt_dir = os.path.join(out_root, 'gt')
    pred_raw_dir = os.path.join(out_root, 'pred_raw')
    gt_raw_dir = os.path.join(out_root, 'gt_raw')

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(pred_raw_dir, exist_ok=True)
    os.makedirs(gt_raw_dir, exist_ok=True)

    image = Image.open(image_path).convert('RGB')
    image.save(os.path.join(image_dir, name + '.jpg'))

    pred_color = get_color_pallete(pred_np.copy(), 'voc')
    pred_color.save(os.path.join(pred_dir, name + '.png'))

    gt_vis = gt_np.copy().astype(np.int32)
    gt_vis[gt_vis == -1] = 255
    gt_color = get_color_pallete(gt_vis.astype(np.int32), 'voc')
    gt_color.save(os.path.join(gt_dir, name + '.png'))

    Image.fromarray(pred_np.astype(np.uint8)).save(os.path.join(pred_raw_dir, name + '.png'))
    Image.fromarray(gt_vis.astype(np.uint8)).save(os.path.join(gt_raw_dir, name + '.png'))


def main():
    args = parse_args()

    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    if use_cuda:
        cudnn.benchmark = True
    device = torch.device('cuda' if use_cuda else 'cpu')

    dataset = VOCDataValSet(args.data, args.data_list)
    loader = data.DataLoader(dataset, batch_size=1, shuffle=False,
                             num_workers=args.workers, pin_memory=use_cuda)

    model = build_model(args, dataset.num_class, device)
    metric = SegmentationMetric(dataset.num_class)

    os.makedirs(args.save_dir, exist_ok=True)

    print('Start VOC testing')
    print('Fixed list:', args.data_list)
    print('Data root :', args.data)
    print('Save dir  :', args.save_dir)
    print('Samples   :', len(loader))

    with torch.no_grad():
        for idx, (image, target, filename) in enumerate(loader, start=1):
            image = image.to(device)
            target = target.long().to(device)

            outputs = model(image)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]

            h, w = target.shape[-2:]
            if outputs.shape[-2:] != (h, w):
                outputs = F.interpolate(outputs, size=(h, w), mode='bilinear', align_corners=True)

            metric.update(outputs, target)
            pixAcc, mIoU = metric.get()

            pred = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()
            gt = target.squeeze(0).cpu().numpy()

            image_path = filename[0][0]
            name = filename[1][0]
            save_outputs(image_path, pred, gt, name, args.save_dir)

            if idx % 20 == 0 or idx == len(loader):
                print('Progress: {}/{} | pixAcc: {:.4f} | mIoU: {:.4f}'.format(idx, len(loader), pixAcc, mIoU))

    pixAcc, mIoU = metric.get()
    print('================ Final Test Result ================')
    print('mIoU: {:.4f}'.format(mIoU))
    print('pixAcc: {:.4f}'.format(pixAcc))
    print('Saved image/pred/gt to:', args.save_dir)


if __name__ == '__main__':
    main()
