import argparse
import time
import datetime
import os
import shutil
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from PCOS import get_max_confidence_and_residual_variance, _class_assignment, _compute_class_centers

from losses import *
from losses import SegCrossEntropyLoss, CriterionKD, CriterionMiniBatchCrossImagePair
from losses import StudentSegContrast, StudentSegChannelContrast
from models.model_zoo import get_segmentation_model

from utils.distributed import *
from utils.logger import setup_logger
from utils.score import SegmentationMetric
from dataset.cityscapes import CSTrainValSet
from dataset.ade20k import ADETrainSet, ADEDataValSet
from dataset.camvid import CamvidTrainSet, CamvidValSet
from dataset.voc import VOCDataTrainSet, VOCDataValSet
from dataset.coco_stuff_164k import CocoStuff164kTrainSet, CocoStuff164kValSet
from utils.flops import cal_multi_adds, cal_param_size

try:
    from scripts.visualize.plot_varg_r_scatter import save_varg_r_scatter
except Exception:
    save_varg_r_scatter = None

try:
    from scripts.visualize.plot_z_r_relation import save_z_r_relation_plot
except Exception:
    save_z_r_relation_plot = None

try:
    from scripts.visualize.plot_g2_r_over_t2 import save_g2_r_over_t2_scatter
except Exception:
    save_g2_r_over_t2_scatter = None


def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Training With Pytorch')
    # model and dataset
    parser.add_argument('--teacher-model', type=str, default='deeplabv3',
                        help='model name')  
    parser.add_argument('--student-model', type=str, default='deeplabv3',
                        help='model name')                      
    parser.add_argument('--student-backbone', type=str, default='resnet18',
                        help='backbone name')
    parser.add_argument('--teacher-backbone', type=str, default='resnet101',
                        help='backbone name')
    parser.add_argument('--dataset', type=str, default='voc',
                        help='dataset name')
    parser.add_argument('--data', type=str, default='./dataset/VOCAug/',  
                        help='dataset directory')
    parser.add_argument('--crop-size', type=int, default=[512, 1024], nargs='+',
                        help='crop image size: [height, width]')
    parser.add_argument('--workers', '-j', type=int, default=8,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--ignore-label', type=int, default=-1, metavar='N',
                        help='ignore label')
    
    # training hyper params
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxiliary loss')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--max-iterations', type=int, default=40000, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.02, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='M',
                        help='w-decay (default: 5e-4)')
    parser.add_argument('--optimizer-type', type=str, default='sgd',
                        help='optimizer type')

    parser.add_argument('--pixel-memory-size', type=int, default=20000)
    parser.add_argument('--region-memory-size', type=int, default=2000)
    parser.add_argument('--channel-memory-size', type=int, default=10000)

    parser.add_argument('--region-contrast-size', type=int, default=1024)
    parser.add_argument('--pixel-contrast-size', type=int, default=4096)
    parser.add_argument('--channel-contrast-size', type=int, default=2048)


    parser.add_argument("--kd-temperature", type=float, default=1.0, help="logits KD temperature")
    parser.add_argument("--contrast-kd-temperature", type=float, default=1.0, help="similarity distribution KD temperature")
    parser.add_argument("--contrast-temperature", type=float, default=0.1, help="similarity distribution temperature")

    parser.add_argument('--use-covar', action='store_true', default=False,
                        help='enable CoVar weighting on teacher outputs')
    parser.add_argument('--covar-temp-base', type=float, default=1.0,
                        help='base temperature T0 for dynamic CoVar KD')
    parser.add_argument('--covar-temp-alpha', type=float, default=0.5,
                        help='alpha for temperature ramp T(x)=ΔT(x)=1+α⋅sqrt(r~(x))')
    parser.add_argument('--covar-temp-min', type=float, default=0.8,
                        help='minimum temperature for dynamic CoVar KD')
    parser.add_argument('--covar-temp-max', type=float, default=2.0,
                        help='maximum temperature for dynamic CoVar KD')
    parser.add_argument('--noise-plot-interval', type=int, default=10000,
                        help='interval (steps) to save z-r diagnostic curve')
    parser.add_argument('--noise-mc-times', type=int, default=20,
                        help='MC dropout forward times for logit noise energy z')
    parser.add_argument('--noise-plot-samples', type=int, default=1000,
                        help='number of (z, r) pairs sampled for each diagnostic plot')
    parser.add_argument('--noise-single-image-pixels', type=int, default=-1,
                        help='number of valid pixels sampled from one selected validation image for z-r diagnostics; <=0 means use all valid pixels')
    parser.add_argument('--noise-single-image-seed', type=int, default=1234,
                        help='random seed for selecting one validation image and pixel sampling')
    parser.add_argument('--noise-min-delta-p2', type=float, default=1e-3,
                        help='minimum ||delta p||^2 threshold for z-r diagnostic sampling; points below are dropped')
    parser.add_argument('--zstar-label-eps', type=float, default=1e-4,
                        help='label smoothing epsilon used to build ideal p_T^* and z_T^*=log p_T^* for z-r diagnostics')
    parser.add_argument('--gradvar-plot-interval', type=int, default=10000,
                        help='interval (steps) to save var(g)-r scatter')
    parser.add_argument('--gradvar-plot-samples', type=int, default=1000,
                        help='number of quantile-sampled points for each var(g)-r scatter')
    parser.add_argument('--g2-rtt-plot-interval', type=int, default=10000,
                        help='interval (steps) to save g^2 vs r/T^2 scatter')
    parser.add_argument('--g2-rtt-plot-samples', type=int, default=1000,
                        help='number of random sampled points for each g^2 vs r/T^2 scatter')
    
    parser.add_argument("--lambda-kd", type=float, default=1., help="lambda_kd")
    parser.add_argument("--lambda-fitnet", type=float, default=0., help="lambda_fitnet")
    parser.add_argument("--lambda-channel-kd", type=float, default=0., help="lambda channel-kd")
    parser.add_argument("--lambda-minibatch-pixel", type=float, default=1., help="lambda mini-batch-based pixel")
    parser.add_argument("--lambda-minibatch-channel", type=float, default=1., help="lambda mini-batch-based channel")
    parser.add_argument("--lambda-memory-pixel", type=float, default=0.1, help="lambda memory-based pixel")
    parser.add_argument("--lambda-memory-region", type=float, default=0.1, help="lambda memory-based region")
    parser.add_argument("--lambda-memory-channel", type=float, default=0.1, help="lambda memory-based channel")
    
    # cuda setting
    parser.add_argument('--gpu-id', type=str, default='0') 
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--local-rank', type=int, default=0)
    # checkpoint and log
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--save-dir', default='~/.torch/models',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--save-dir-name', default='seg_kd_exps',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--save-epoch', type=int, default=10,
                        help='save model every checkpoint-epoch')
    parser.add_argument('--log-iter', type=int, default=10,
                        help='print log every log-iter')
    parser.add_argument('--save-per-iters', type=int, default=800,
                        help='per iters to save')
    parser.add_argument('--val-per-iters', type=int, default=800,
                        help='per iters to val')
    parser.add_argument('--topk-checkpoints', type=int, default=5,
                        help='keep top-K checkpoints by validation mIoU')
    parser.add_argument('--teacher-pretrained-base', type=str, default='None',
                        help='pretrained backbone')
    parser.add_argument('--teacher-pretrained', type=str, default='None',
                        help='pretrained seg model')
    parser.add_argument('--student-pretrained-base', type=str, default='None',
                    help='pretrained backbone')
    parser.add_argument('--student-pretrained', type=str, default='None',
                        help='pretrained seg model')

                        
    # evaluation only
    parser.add_argument('--val-epoch', type=int, default=1,
                        help='run validation every val-epoch')
    parser.add_argument('--skip-val', action='store_true', default=False,
                        help='skip validation during training')
    args = parser.parse_args()
    # Allow torchrun to supply LOCAL_RANK via env (torch.distributed.launch passes --local_rank)
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    args.save_dir = os.path.join(args.save_dir, args.save_dir_name)
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    if num_gpus > 1 and args.local_rank == 0:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    if args.student_backbone.startswith('resnet'):
        args.aux = True
    else:
        args.aux = False

    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args
        if args.distributed and args.device == "cuda":
            self.device = torch.device("cuda", args.local_rank)
        else:
            self.device = torch.device(args.device)
        self.num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

        if args.dataset == 'citys':
            train_dataset = CSTrainValSet(args.data, 
                                            list_path='./dataset/list/cityscapes/train.lst', 
                                            max_iters=args.max_iterations*args.batch_size, 
                                            crop_size=args.crop_size, scale=True, mirror=True)
            val_dataset = CSTrainValSet(args.data, 
                                        list_path='./dataset/list/cityscapes/val.lst', 
                                        crop_size=(1024, 2048), scale=False, mirror=False)
        elif args.dataset == 'voc':
            train_dataset = VOCDataTrainSet(args.data, './dataset/list/voc/train_aug.txt', max_iters=args.max_iterations*args.batch_size, 
                                          crop_size=args.crop_size, scale=True, mirror=True)
            val_dataset = VOCDataValSet(args.data, './dataset/list/voc/val.txt')
        elif args.dataset == 'ade20k':
            train_dataset = ADETrainSet(args.data, max_iters=args.max_iterations*args.batch_size, ignore_label=args.ignore_label,
                                        crop_size=args.crop_size, scale=True, mirror=True)
            val_dataset = ADEDataValSet(args.data)
        elif args.dataset == 'camvid':
            train_dataset = CamvidTrainSet(args.data, './dataset/list/CamVid/camvid_train_list.txt', max_iters=args.max_iterations*args.batch_size,
                            ignore_label=args.ignore_label, crop_size=args.crop_size, scale=True, mirror=True)
            val_dataset = CamvidValSet(args.data, './dataset/list/CamVid/camvid_val_list.txt')
        elif args.dataset == 'coco_stuff_164k':
            train_dataset = CocoStuff164kTrainSet(args.data, './dataset/list/coco_stuff_164k/coco_stuff_164k_train.txt', max_iters=args.max_iterations*args.batch_size, ignore_label=args.ignore_label,
                                        crop_size=args.crop_size, scale=True, mirror=True)
            val_dataset = CocoStuff164kValSet(args.data, './dataset/list/coco_stuff_164k/coco_stuff_164k_val.txt')
        else:
            raise ValueError('dataset unfind')

    
        args.batch_size = args.batch_size // num_gpus
        train_sampler = make_data_sampler(train_dataset, shuffle=True, distributed=args.distributed)
        train_batch_sampler = make_batch_data_sampler(train_sampler, args.batch_size, args.max_iterations)
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=1)

        self.train_loader = data.DataLoader(dataset=train_dataset,
                                            batch_sampler=train_batch_sampler,
                                            num_workers=args.workers,
                                            pin_memory=True)

        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=args.workers,
                                          pin_memory=True)

        # create network
        BatchNorm2d = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d

        self.t_model = get_segmentation_model(model=args.teacher_model, 
                                            backbone=args.teacher_backbone,
                                            local_rank=args.local_rank,
                                            pretrained_base='None',
                                            pretrained=args.teacher_pretrained,
                                            aux=True, 
                                            norm_layer=nn.BatchNorm2d,
                            num_class=train_dataset.num_class).to(self.device)

        self.s_model = get_segmentation_model(model=args.student_model, 
                                            backbone=args.student_backbone,
                                            local_rank=args.local_rank,
                                            pretrained_base=args.student_pretrained_base,
                                            pretrained='None',
                                            aux=args.aux, 
                                            norm_layer=BatchNorm2d,
                                            num_class=train_dataset.num_class).to(self.device)
        
        for t_n, t_p in self.t_model.named_parameters():
            t_p.requires_grad = False
        self.t_model.eval()
        self.s_model.eval()


        # resume checkpoint if needed
        if args.resume:
            if os.path.isfile(args.resume):
                name, ext = os.path.splitext(args.resume)
                assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
                print('Resuming training, loading {}...'.format(args.resume))
                self.s_model.load_state_dict(torch.load(args.resume, map_location=lambda storage, loc: storage))

        # create criterion
        x = torch.randn(1, 3, args.crop_size[0], args.crop_size[1]).to(self.device)
        t_y = self.t_model(x)
        s_y = self.s_model(x)
        t_size = t_y[-1].size()
        s_size = s_y[-1].size()
        t_channels = t_size[1]
        s_channels = s_size[1]

        self.criterion = SegCrossEntropyLoss(ignore_index=args.ignore_label).to(self.device)
        self.criterion_kd = CriterionKD(temperature=args.kd_temperature).to(self.device)
        self.criterion_fitnet = CriterionFitNet(s_channels, t_channels).to(self.device)
        self.gcn = GCN(s_channels, BatchNorm2d).to(self.device)
        self.criterion_minibatch = CriterionMiniBatchCrossImagePair(temperature=args.contrast_temperature).to(self.device)
        self.criterion_memory_contrast = StudentSegContrast(num_classes=train_dataset.num_class,
                                                     pixel_memory_size=args.pixel_memory_size,
                                                     region_memory_size=args.region_memory_size,
                                                     region_contrast_size=args.region_contrast_size//train_dataset.num_class+1,
                                                     pixel_contrast_size=args.pixel_contrast_size//train_dataset.num_class+1,
                                                     contrast_kd_temperature=args.contrast_kd_temperature,
                                                     contrast_temperature=args.contrast_temperature,
                                                     s_channels=s_channels,
                                                     t_channels=t_channels,
                                                     ignore_label=args.ignore_label).to(self.device)

        self.criterion_channel_contrast = StudentSegChannelContrast(channel_memory_size=args.channel_memory_size, 
                                                    channel_contrast_size=args.channel_contrast_size, 
                                                    contrast_kd_temperature=args.contrast_kd_temperature, 
                                                    contrast_temperature=args.contrast_temperature,
                                                    s_size=s_size,
                                                    t_size=t_size,
                                                    ).to(self.device)
    
        params_list = nn.ModuleList([])
        params_list.append(self.s_model)
        params_list.append(self.criterion_memory_contrast)
        params_list.append(self.criterion_fitnet)
        params_list.append(self.criterion_channel_contrast)
        params_list.append(self.gcn)
        
        
        if args.optimizer_type == 'sgd':
            self.optimizer = torch.optim.SGD(params_list.parameters(),
                                            lr=args.lr,
                                            momentum=args.momentum,
                                            weight_decay=args.weight_decay)
        elif args.optimizer_type == 'adamw':
            self.optimizer = torch.optim.AdamW(params_list.parameters(),
                                               lr=args.lr,
                                               weight_decay=args.weight_decay)
        else:
            raise ValueError('no such optimizer')


        if args.distributed:
            self.s_model = nn.parallel.DistributedDataParallel(self.s_model, 
                                                                device_ids=[args.local_rank],
                                                                output_device=args.local_rank)
            self.criterion_memory_contrast = nn.parallel.DistributedDataParallel(self.criterion_memory_contrast, 
                                                             device_ids=[args.local_rank],
                                                             output_device=args.local_rank)
            self.criterion_channel_contrast = nn.parallel.DistributedDataParallel(self.criterion_channel_contrast, 
                                                             device_ids=[args.local_rank],
                                                             output_device=args.local_rank)
            
        # evaluation metrics
        self.metric = SegmentationMetric(train_dataset.num_class)
        self.best_pred = 0.0
        # track temperature statistics for logging and plotting
        self.temp_history = []
        # track top-K checkpoints (mIoU, filepath)
        self.topk_checkpoints = []
        self.topk_k = args.topk_checkpoints
        self.val_delta_z2_buffer = []
        self.val_delta_p2_buffer = []
        self.val_r_buffer = []
        self.grad2_buffer = []
        self.gradvar_buffer = []
        self.gradvar_r_buffer = []
        self.g2_rtt_g2_buffer = []
        self.g2_rtt_y_buffer = []

    def adjust_lr(self, base_lr, iter, max_iter, power):
        cur_lr = base_lr*((1-float(iter)/max_iter)**(power))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = cur_lr

        return cur_lr

    def reduce_tensor(self, tensor):
        if not dist.is_available() or not dist.is_initialized() or self.num_gpus == 1:
            return tensor
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        return rt

    def reduce_mean_tensor(self, tensor):
        if not dist.is_available() or not dist.is_initialized() or self.num_gpus == 1:
            return tensor
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.num_gpus
        return rt

    @torch.no_grad()
    def split_quality(self, max_confidence, scaled_residual_variance, valid_mask):
        n, h, w = max_confidence.shape
        mask_high = torch.zeros_like(valid_mask, dtype=torch.bool)
        for i in range(n):
            valid_flat = valid_mask[i].view(-1)
            if valid_flat.sum() == 0:
                continue
            feats = torch.stack([max_confidence[i], scaled_residual_variance[i]], dim=-1).view(-1, 2)
            feats_valid = feats[valid_flat]
            assignments = _class_assignment(feats_valid, 2)
            means, _ = _compute_class_centers(feats_valid, assignments, 2)
            high_label = torch.argmax(means[:, 0]).item()
            assign_full = torch.zeros_like(valid_flat, dtype=torch.long)
            assign_full[valid_flat] = assignments
            mask_high[i] = (assign_full == high_label).view(h, w) & valid_mask[i]
        return mask_high

    @torch.no_grad()
    def get_covar_metadata(self, teacher_logits, valid_mask, epsilon=1e-8):
        _, _, ht, wt = teacher_logits.shape
        if valid_mask.shape[-2:] != (ht, wt):
            vm = valid_mask.float().unsqueeze(1)
            vm = F.interpolate(vm, size=(ht, wt), mode='nearest')
            valid_mask_resized = (vm.squeeze(1) > 0.5)
        else:
            valid_mask_resized = valid_mask

        prob = F.softmax(teacher_logits, dim=1)
        num_classes = prob.size(1)

        max_confidence, scaled_residual_variance = \
            get_max_confidence_and_residual_variance(
                prob, valid_mask_resized, num_classes, epsilon=epsilon
            )

        mask_high = self.split_quality(
            max_confidence, scaled_residual_variance, valid_mask_resized
        )

        # ==========================================
        # NEW: sqrt-reliability temperature (theory)
        # ==========================================

        r = scaled_residual_variance  # r(x)

        if valid_mask_resized.any():
            r_valid = r[valid_mask_resized]
            r_mean = r_valid.mean()
        else:
            r_mean = r.mean()

        # ---- normalize reliability scale ----
        # r_norm = r / (r_mean + epsilon)

        # ---- critical-order transform ----
        sqrt_r = torch.sqrt(r+ epsilon)

        # ---- zero-mean constraint ----
        if valid_mask_resized.any():
            sqrt_r_mean = sqrt_r[valid_mask_resized].mean().detach()
        else:
            sqrt_r_mean = sqrt_r.mean()

        sqrt_r_centered = sqrt_r - sqrt_r_mean

        # ---- modulation strength (same role as before) ----
        alpha = self.args.covar_temp_alpha  # e.g. 0.5

        delta_T = alpha * sqrt_r_centered

        # base temperature = 1
        temp_map = 1.0 + delta_T

        # ---- safety clamp (rarely triggered) ----
        T_min = getattr(self.args, "covar_temp_min", 0.7)
        T_max = getattr(self.args, "covar_temp_max", 1.7)
        temp_map = torch.clamp(temp_map, min=T_min, max=T_max)

        # ---- logging ----
        if valid_mask_resized.any():
            temp_valid = temp_map[valid_mask_resized]
            temp_mean = temp_valid.mean().item()
            temp_max = temp_valid.max().item()
            temp_min = temp_valid.min().item()
        else:
            temp_mean = temp_map.mean().item()
            temp_max = temp_map.max().item()
            temp_min = temp_map.min().item()

        self.temp_history.append((temp_mean, temp_min, temp_max))

        return temp_map, mask_high, valid_mask_resized, r


    def covar_temperature_kd_loss(self, student_logits, teacher_logits, temperature_map, valid_mask, epsilon=1e-8):
        # Broadcast per-pixel temperature across channel dimension
        temp = temperature_map.unsqueeze(1)
        s_log_prob = F.log_softmax(student_logits / temp, dim=1)
        t_prob = F.softmax(teacher_logits / temp, dim=1)

        kd_map = F.kl_div(s_log_prob, t_prob, reduction='none').sum(dim=1)
        denom = valid_mask.sum().clamp_min(1)
        loss = kd_map.sum() / denom
        return loss

    @staticmethod
    def _extract_main_logits(outputs):
        if isinstance(outputs, (list, tuple)):
            return outputs[0]
        return outputs

    def _gather_1d_from_all_ranks(self, x_1d):
        if not (dist.is_available() and dist.is_initialized() and self.num_gpus > 1):
            return x_1d.detach().float().cpu()

        x_1d = x_1d.contiguous()
        local_len = torch.tensor([x_1d.numel()], device=x_1d.device, dtype=torch.long)
        len_gather = [torch.zeros_like(local_len) for _ in range(self.num_gpus)]
        dist.all_gather(len_gather, local_len)
        lens = [int(t.item()) for t in len_gather]
        max_len = max(lens) if lens else 0

        if max_len == 0:
            if get_rank() == 0:
                return torch.empty(0, dtype=torch.float32)
            return None

        pad = torch.zeros(max_len, device=x_1d.device, dtype=x_1d.dtype)
        if x_1d.numel() > 0:
            pad[:x_1d.numel()] = x_1d

        gather_pad = [torch.zeros_like(pad) for _ in range(self.num_gpus)]
        dist.all_gather(gather_pad, pad)

        if get_rank() != 0:
            return None

        out = []
        for i, g in enumerate(gather_pad):
            if lens[i] > 0:
                out.append(g[:lens[i]].detach().float().cpu())
        if not out:
            return torch.empty(0, dtype=torch.float32)
        return torch.cat(out, dim=0)

    @torch.no_grad()
    def _build_ideal_prob_and_logits_from_labels(self, targets, num_classes):
        """
        Build p_T^* from labels and derive one admissible z_T^*=log(p_T^*) so that
        softmax(z_T^*) = p_T^* exactly. We use epsilon smoothing to avoid log(0).
        """
        eps = float(self.args.zstar_label_eps)
        eps = min(max(eps, 1e-8), 1.0 - 1e-6)

        targets_clamped = targets.clone()
        targets_clamped[targets_clamped < 0] = 0

        onehot = F.one_hot(targets_clamped, num_classes=num_classes).permute(0, 3, 1, 2).float()
        neg = eps / max(num_classes - 1, 1)
        pos = 1.0 - eps
        p_star = onehot * pos + (1.0 - onehot) * neg

        # z_T^*=log p_T^* is defined up to additive constant per pixel.
        z_star = torch.log(p_star.clamp_min(1e-12))
        z_star = z_star - z_star.mean(dim=1, keepdim=True)
        return p_star, z_star

    @torch.no_grad()
    def collect_val_z_dp_r_pairs(self, teacher_logits, targets, epsilon=1e-8):
        valid_mask = (targets != self.args.ignore_label)
        if not valid_mask.any():
            return

        prob = F.softmax(teacher_logits, dim=1)
        num_classes = prob.size(1)
        _, r_map = get_max_confidence_and_residual_variance(
            prob, valid_mask, num_classes, epsilon=epsilon
        )

        p_star, z_star = self._build_ideal_prob_and_logits_from_labels(targets, num_classes)

        delta_z = teacher_logits - z_star
        # Jacobian-vector product: J(p)v = diag(p)v - p(p^T v)
        p_dot_dz = (prob * delta_z).sum(dim=1, keepdim=True)
        delta_p = prob * (delta_z - p_dot_dz)

        delta_z2 = (delta_z ** 2).sum(dim=1)
        delta_p2 = (delta_p ** 2).sum(dim=1)

        z_valid = delta_z2[valid_mask]
        dp_valid = delta_p2[valid_mask]
        r_valid = r_map[valid_mask]

        min_dp2 = float(self.args.noise_min_delta_p2)
        keep = (
            torch.isfinite(z_valid)
            & torch.isfinite(dp_valid)
            & torch.isfinite(r_valid)
            & (dp_valid >= 0.0)
            & (dp_valid >= min_dp2)
        )
        if keep.any():
            z_valid = z_valid[keep]
            dp_valid = dp_valid[keep]
            r_valid = r_valid[keep]
        else:
            return

        max_pixels = int(self.args.noise_single_image_pixels)
        n_valid = z_valid.numel()
        if n_valid == 0:
            return
        if max_pixels > 0 and n_valid > max_pixels:
            idx = torch.randperm(n_valid, device=z_valid.device)[:max_pixels]
            z_valid = z_valid[idx]
            dp_valid = dp_valid[idx]
            r_valid = r_valid[idx]

        self.val_delta_z2_buffer.append(z_valid.detach().float().cpu())
        self.val_delta_p2_buffer.append(dp_valid.detach().float().cpu())
        self.val_r_buffer.append(r_valid.detach().float().cpu())

    @torch.no_grad()
    def save_noise_r_diagnostic(self, step):
        if self.val_delta_z2_buffer:
            delta_z2_local = torch.cat(self.val_delta_z2_buffer, dim=0).to(self.device)
        else:
            delta_z2_local = torch.empty(0, device=self.device, dtype=torch.float32)

        if self.val_delta_p2_buffer:
            delta_p2_local = torch.cat(self.val_delta_p2_buffer, dim=0).to(self.device)
        else:
            delta_p2_local = torch.empty(0, device=self.device, dtype=torch.float32)

        if self.val_r_buffer:
            r_local = torch.cat(self.val_r_buffer, dim=0).to(self.device)
        else:
            r_local = torch.empty(0, device=self.device, dtype=torch.float32)

        z_cpu = self._gather_1d_from_all_ranks(delta_z2_local)
        dp_cpu = self._gather_1d_from_all_ranks(delta_p2_local)
        r_cpu = self._gather_1d_from_all_ranks(r_local)

        self.val_delta_z2_buffer = []
        self.val_delta_p2_buffer = []
        self.val_r_buffer = []

        if (dist.is_available() and dist.is_initialized() and self.num_gpus > 1) and get_rank() != 0:
            return

        if z_cpu is None or dp_cpu is None or r_cpu is None or z_cpu.numel() == 0:
            logger.info(f"Skipping z-r diagnostic at step {step}: gathered pairs are empty")
            return

        pairs_path = os.path.join(self.args.save_dir, f'z_r_pairs_step_{step}.pt')
        try:
            torch.save({'delta_z_norm2': z_cpu, 'delta_p_norm2': dp_cpu, 'r': r_cpu, 'step': step}, pairs_path)
            logger.info(f"Saved z-r pairs to {pairs_path}")
        except Exception as e:
            logger.info(f"Failed to save z-r pairs at step {step}: {e}")

        out_path = os.path.join(self.args.save_dir, f'z_r_curve_step_{step}.png')

        if save_z_r_relation_plot is None:
            logger.info("Skipping z-r diagnostic plot: plotting script import failed")
            return

        try:
            sample_num = int(z_cpu.numel())
            if int(self.args.noise_single_image_pixels) > 0:
                sample_num = min(sample_num, int(self.args.noise_single_image_pixels))
            stats = save_z_r_relation_plot(
                z_cpu,
                dp_cpu,
                r_cpu,
                out_path,
                sample_num=max(sample_num, 1),
                title=f'Relation: ||delta z||^2 vs ||delta p||^2 and r @ step {step}'
            )
            fit_dp = stats.get('fit_delta_p2', None)
            fit_r = stats.get('fit_r', None)
            msg = f"Saved z-r diagnostic to {out_path} with {stats['num_points']} random sampled points"
            if fit_dp is not None:
                msg += f"; dp2 corr={fit_dp['corr']:.4f}, R^2={fit_dp['r2']:.4f}"
            if fit_r is not None:
                msg += f"; r corr={fit_r['corr']:.4f}, R^2={fit_r['r2']:.4f}"
            logger.info(msg)
        except Exception as e:
            logger.info(f"Failed to save z-r diagnostic at step {step}: {e}")

    @torch.no_grad()
    def collect_gradvar_r_pairs(self, student_logits, teacher_logits, temperature_map, r_map, valid_mask):
        temp = temperature_map.unsqueeze(1).clamp_min(1e-8)
        ps = F.softmax(student_logits / temp, dim=1)
        pt = F.softmax(teacher_logits / temp, dim=1)
        g = (ps - pt) / temp
        g2 = (g ** 2).mean(dim=1)
        var_g = torch.var(g, dim=1, unbiased=False)

        g2_valid = g2[valid_mask]
        var_valid = var_g[valid_mask]
        r_valid = r_map[valid_mask]
        if var_valid.numel() == 0:
            return

        self.grad2_buffer.append(g2_valid.detach().float().cpu())
        self.gradvar_buffer.append(var_valid.detach().float().cpu())
        self.gradvar_r_buffer.append(r_valid.detach().float().cpu())

    @torch.no_grad()
    def save_gradvar_r_scatter(self, step):
        if not self.grad2_buffer or not self.gradvar_buffer or not self.gradvar_r_buffer:
            logger.info(f"Skipping var(g)-r scatter at step {step}: no collected pairs")
            return

        g2_cpu = torch.cat(self.grad2_buffer, dim=0)
        var_g_cpu = torch.cat(self.gradvar_buffer, dim=0)
        r_cpu = torch.cat(self.gradvar_r_buffer, dim=0)

        pairs_path = os.path.join(self.args.save_dir, f'varg_r_pairs_step_{step}.pt')
        try:
            torch.save({'g2': g2_cpu, 'var_g': var_g_cpu, 'r': r_cpu}, pairs_path)
            logger.info(f"Saved var(g)-r pairs to {pairs_path}")
        except Exception as e:
            logger.info(f"Failed to save var(g)-r pairs at step {step}: {e}")

        out_path = os.path.join(self.args.save_dir, f'varg_r_scatter_step_{step}.png')
        if save_varg_r_scatter is None:
            logger.info("Skipping var(g)-r scatter: plotting script import failed")
            self.grad2_buffer = []
            self.gradvar_buffer = []
            self.gradvar_r_buffer = []
            return

        try:
            stats = save_varg_r_scatter(
                g2_cpu,
                var_g_cpu,
                r_cpu,
                out_path,
                sample_num=self.args.gradvar_plot_samples,
                title=f'r-g^2-var(g) Scatter @ step {step}'
            )
            logger.info(
                f"Saved var(g)-r scatter to {out_path} with {stats['num_points']} quantile-sampled points; "
                f"corr={stats['corr']:.6f}, mean|g^2-var|={stats['mean_abs_diff']:.3e}, "
                f"max|g^2-var|={stats['max_abs_diff']:.3e}"
            )
        except Exception as e:
            logger.info(f"Failed to save var(g)-r scatter at step {step}: {e}")

        self.grad2_buffer = []
        self.gradvar_buffer = []
        self.gradvar_r_buffer = []

    @torch.no_grad()
    def collect_g2_r_over_t2_pairs(self, student_logits, teacher_logits, temperature_map, r_map, valid_mask):
        temp = temperature_map.unsqueeze(1).clamp_min(1e-8)
        ps = F.softmax(student_logits / temp, dim=1)
        pt = F.softmax(teacher_logits / temp, dim=1)
        g = (ps - pt) / temp
        g2 = (g ** 2).mean(dim=1)
        r_over_t2 = r_map / (temperature_map.clamp_min(1e-8) ** 2)

        g2_valid = g2[valid_mask]
        y_valid = r_over_t2[valid_mask]
        if g2_valid.numel() == 0:
            return

        self.g2_rtt_g2_buffer.append(g2_valid.detach().float().cpu())
        self.g2_rtt_y_buffer.append(y_valid.detach().float().cpu())

    @torch.no_grad()
    def save_g2_r_over_t2_scatter(self, step):
        if not self.g2_rtt_g2_buffer or not self.g2_rtt_y_buffer:
            logger.info(f"Skipping g^2-r/T^2 scatter at step {step}: no collected pairs")
            return

        g2_cpu = torch.cat(self.g2_rtt_g2_buffer, dim=0)
        y_cpu = torch.cat(self.g2_rtt_y_buffer, dim=0)

        pairs_path = os.path.join(self.args.save_dir, f'g2_r_over_t2_pairs_step_{step}.pt')
        try:
            torch.save({'g2': g2_cpu, 'r_over_t2': y_cpu, 'step': step}, pairs_path)
            logger.info(f"Saved g^2-r/T^2 pairs to {pairs_path}")
        except Exception as e:
            logger.info(f"Failed to save g^2-r/T^2 pairs at step {step}: {e}")

        out_path = os.path.join(self.args.save_dir, f'g2_r_over_t2_scatter_step_{step}.png')
        if save_g2_r_over_t2_scatter is None:
            logger.info("Skipping g^2-r/T^2 scatter: plotting script import failed")
            self.g2_rtt_g2_buffer = []
            self.g2_rtt_y_buffer = []
            return

        try:
            stats = save_g2_r_over_t2_scatter(
                g2_cpu,
                y_cpu,
                out_path,
                sample_num=self.args.g2_rtt_plot_samples,
                title=f'g^2 vs r/T^2 @ step {step}'
            )
            logger.info(
                f"Saved g^2-r/T^2 scatter to {out_path} with {stats['num_points']} random sampled points; "
                f"corr={stats['corr']:.6f}"
            )
        except Exception as e:
            logger.info(f"Failed to save g^2-r/T^2 scatter at step {step}: {e}")

        self.g2_rtt_g2_buffer = []
        self.g2_rtt_y_buffer = []

    def train(self):
        save_to_disk = get_rank() == 0
        log_per_iters, val_per_iters = self.args.log_iter, self.args.val_per_iters
        save_per_iters = self.args.save_per_iters
        start_time = time.time()
        logger.info('Start training, Total Iterations {:d}'.format(args.max_iterations))

        self.s_model.train()
        for iteration, (images, targets, _) in enumerate(self.train_loader):
            iteration = iteration + 1
            
            images = images.to(self.device)
            targets = targets.long().to(self.device)
            
            with torch.no_grad():
                t_outputs = self.t_model(images)

            s_outputs = self.s_model(images)

            temperature_map = None
            r_map = None
            valid_mask_resized = None
            need_covar_metadata = self.args.use_covar and (
                self.args.lambda_kd != 0.
                or (save_to_disk and self.args.gradvar_plot_interval > 0)
                or (save_to_disk and self.args.g2_rtt_plot_interval > 0)
            )
            if need_covar_metadata:
                with torch.no_grad():
                    valid_mask = (targets != self.args.ignore_label)
                    temperature_map, _, valid_mask_resized, r_map = self.get_covar_metadata(t_outputs[0], valid_mask)

                if save_to_disk and self.args.gradvar_plot_interval > 0:
                    self.collect_gradvar_r_pairs(
                        s_outputs[0], t_outputs[0], temperature_map, r_map, valid_mask_resized
                    )
                    if iteration % self.args.gradvar_plot_interval == 0:
                        self.save_gradvar_r_scatter(iteration)

                if save_to_disk and self.args.g2_rtt_plot_interval > 0:
                    self.collect_g2_r_over_t2_pairs(
                        s_outputs[0], t_outputs[0], temperature_map, r_map, valid_mask_resized
                    )
                    if iteration % self.args.g2_rtt_plot_interval == 0:
                        self.save_g2_r_over_t2_scatter(iteration)
            
            if self.args.aux:
                task_loss = self.criterion(s_outputs[0], targets) + 0.4 * self.criterion(s_outputs[1], targets)
            else:
                task_loss = self.criterion(s_outputs[0], targets)
            

            kd_loss = torch.tensor(0.).to(self.device)
            fitnet_loss = torch.tensor(0.).to(self.device)
            
            if self.args.lambda_kd != 0.:
                if self.args.use_covar and temperature_map is not None:
                    kd_loss = self.args.lambda_kd * self.covar_temperature_kd_loss(
                        s_outputs[0], t_outputs[0], temperature_map, valid_mask_resized)
                else:
                    kd_loss = self.args.lambda_kd * self.criterion_kd(s_outputs[0], t_outputs[0])

            if self.args.lambda_fitnet:
                s_outputs[-1] = self.gcn(s_outputs[-1])
                fitnet_loss = self.args.lambda_fitnet * self.criterion_fitnet(s_outputs[-1], t_outputs[-1])


            minibatch_pixel_contrast_loss = \
                self.args.lambda_minibatch_pixel * self.criterion_minibatch(s_outputs[-1], t_outputs[-1])

            _, predict = torch.max(s_outputs[0], dim=1) 
            memory_pixel_contrast_loss, memory_region_contrast_loss = \
                self.criterion_memory_contrast(s_outputs[-1], t_outputs[-1].detach(), targets, predict)
            
            memory_pixel_contrast_loss = self.args.lambda_memory_pixel * memory_pixel_contrast_loss
            memory_region_contrast_loss = self.args.lambda_memory_region * memory_region_contrast_loss

            minibatch_channel_contrast_loss, memory_channel_contrast_loss, channel_mse_loss = \
                self.criterion_channel_contrast(s_outputs[-1], t_outputs[-1].detach())
            minibatch_channel_contrast_loss = args.lambda_minibatch_channel * minibatch_channel_contrast_loss
            memory_channel_contrast_loss =args.lambda_memory_channel * memory_channel_contrast_loss
            channel_mse_loss = args.lambda_channel_kd * channel_mse_loss
              
            losses = task_loss + kd_loss + minibatch_pixel_contrast_loss + \
                memory_pixel_contrast_loss + memory_region_contrast_loss + fitnet_loss + \
                minibatch_channel_contrast_loss + memory_channel_contrast_loss + channel_mse_loss
                
            
            lr = self.adjust_lr(base_lr=args.lr, iter=iteration-1, max_iter=args.max_iterations, power=0.9)
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            task_losses_reduced = self.reduce_mean_tensor(task_loss)
            kd_losses_reduced = self.reduce_mean_tensor(kd_loss)
            minibatch_pixel_contrast_loss_reduced = self.reduce_mean_tensor(minibatch_pixel_contrast_loss)
            memory_pixel_contrast_loss_reduced = self.reduce_mean_tensor(memory_pixel_contrast_loss)
            memory_region_contrast_loss_reduced = self.reduce_mean_tensor(memory_region_contrast_loss)
            minibatch_channel_contrast_loss_reduced = self.reduce_mean_tensor(minibatch_channel_contrast_loss)
            memory_channel_contrast_loss_reduced = self.reduce_mean_tensor(memory_channel_contrast_loss)
            channel_mse_loss_reduced = self.reduce_mean_tensor(channel_mse_loss)
            fitnet_loss_reduced = self.reduce_mean_tensor(fitnet_loss)
            
            
            eta_seconds = ((time.time() - start_time) / iteration) * (args.max_iterations - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % log_per_iters == 0 and save_to_disk:
                t_info = ""
                if self.args.use_covar and self.temp_history:
                    t_mean, t_min, t_max = self.temp_history[-1]
                    if len(self.temp_history) >= 2:
                        prev_mean = self.temp_history[-2][0]
                        d_mean = t_mean - prev_mean
                        t_info = f" || T_mean: {t_mean:.4f} (d{d_mean:+.4f}) || T_min: {t_min:.4f} || T_max: {t_max:.4f}"
                    else:
                        t_info = f" || T_mean: {t_mean:.4f} || T_min: {t_min:.4f} || T_max: {t_max:.4f}"
                logger.info(
                    "Iters: {:d}/{:d} || Lr: {:.6f} || Task Loss: {:.4f} || KD Loss: {:.4f} " \
                    "|| Mini-batch p2p Loss: {:.4f} || Memory p2p Loss: {:.4f} || Memory p2r Loss: {:.4f} " \
                    "|| Mini-batch c2c Loss: {:.4f} || Memory c2c Loss: {:.4f} || Channel MSE Loss: {:.4f} " \
                    "|| Fitnet Loss: {:.4f}{} " \
                    "|| Cost Time: {} || Estimated Time: {}".format(
                        iteration, args.max_iterations, self.optimizer.param_groups[0]['lr'], task_losses_reduced.item(),
                        kd_losses_reduced.item(), 
                        minibatch_pixel_contrast_loss_reduced.item(),
                        memory_pixel_contrast_loss_reduced.item(),
                        memory_region_contrast_loss_reduced.item(),
                        minibatch_channel_contrast_loss_reduced.item(),
                        memory_channel_contrast_loss_reduced.item(),
                        channel_mse_loss_reduced.item(),
                        fitnet_loss_reduced.item(), t_info,
                        str(datetime.timedelta(seconds=int(time.time() - start_time))), eta_string))

            if iteration % save_per_iters == 0 and save_to_disk:
                save_checkpoint(self.s_model, self.args, is_best=False)

            if not self.args.skip_val and iteration % val_per_iters == 0:
                self.validation(step=iteration)
                self.s_model.train()

        save_checkpoint(self.s_model, self.args, is_best=False)
        total_training_time = time.time() - start_time
        total_training_str = str(datetime.timedelta(seconds=total_training_time))
        logger.info(
            "Total training time: {} ({:.4f}s / it)".format(
                total_training_str, total_training_time / args.max_iterations))
        # save temperature curve if CoVar is enabled
        if self.args.use_covar and get_rank() == 0:
            self.save_temperature_curve()
        


    def validation(self, step=None):
        is_best = False
        self.metric.reset()
        self.val_delta_z2_buffer = []
        self.val_delta_p2_buffer = []
        self.val_r_buffer = []
        if self.args.distributed:
            model = self.s_model.module
        else:
            model = self.s_model
        torch.cuda.empty_cache()  # TODO check if it helps
        model.eval()
        logger.info("Start validation, Total sample: {:d}".format(len(self.val_loader)))

        selected_local_idx = None
        if self.args.use_covar:
            rng = torch.Generator(device='cpu')
            seed = int(self.args.noise_single_image_seed) + int(0 if step is None else step)
            rng.manual_seed(seed)
            selected_local_idx = int(torch.randint(low=0, high=len(self.val_loader), size=(1,), generator=rng).item())
            if (not self.args.distributed) or get_rank() == 0:
                logger.info(
                    f"z-r diagnostic selected validation image local index: {selected_local_idx} "
                    f"(seed={seed}, sample_pixels={self.args.noise_single_image_pixels})"
                )

        for i, (image, target, filename) in enumerate(self.val_loader):
            image = image.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                outputs = model(image)
                t_outputs = self.t_model(image)

            B, H, W = target.size()
            outputs[0] = F.interpolate(outputs[0], (H, W), mode='bilinear', align_corners=True)
            t_logits = F.interpolate(t_outputs[0], (H, W), mode='bilinear', align_corners=True)

            if self.args.use_covar and selected_local_idx is not None and i == selected_local_idx:
                if (not self.args.distributed) or get_rank() == 0:
                    logger.info(f"z-r diagnostic picked filename: {filename}")
                    self.collect_val_z_dp_r_pairs(t_logits, target)

            self.metric.update(outputs[0], target)
            pixAcc, mIoU = self.metric.get()
            logger.info("Sample: {:d}, Validation pixAcc: {:.3f}, mIoU: {:.3f}".format(i + 1, pixAcc, mIoU))

        if self.args.use_covar:
            diag_step = 0 if step is None else step
            self.save_noise_r_diagnostic(diag_step)
        
        if self.num_gpus > 1:
            sum_total_correct = torch.tensor(self.metric.total_correct).to(self.device)
            sum_total_label = torch.tensor(self.metric.total_label).to(self.device)
            sum_total_inter = torch.tensor(self.metric.total_inter).to(self.device)
            sum_total_union = torch.tensor(self.metric.total_union).to(self.device)
            sum_total_correct = self.reduce_tensor(sum_total_correct)
            sum_total_label = self.reduce_tensor(sum_total_label)
            sum_total_inter = self.reduce_tensor(sum_total_inter)
            sum_total_union = self.reduce_tensor(sum_total_union)

            pixAcc = 1.0 * sum_total_correct / (2.220446049250313e-16 + sum_total_label) 
            IoU = 1.0 * sum_total_inter / (2.220446049250313e-16 + sum_total_union)
            mIoU = IoU.mean().item()

            logger.info("Overall validation pixAcc: {:.3f}, mIoU: {:.3f}".format(
            pixAcc.item() * 100, mIoU * 100))

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
        if (args.distributed is not True) or (args.distributed and args.local_rank == 0):
            save_checkpoint(self.s_model, self.args, is_best)
            # maintain top-K by mIoU
            self._maybe_save_topk(new_pred)
        synchronize()

    def _maybe_save_topk(self, score):
        try:
            directory = os.path.expanduser(self.args.save_dir)
            if not os.path.exists(directory):
                os.makedirs(directory)
            # save with mIoU in filename
            filename = 'kd_{}_{}_{}_miou-{:.4f}.pth'.format(self.args.student_model, self.args.student_backbone, self.args.dataset, score)
            path = os.path.join(directory, filename)
            model = self.s_model
            if self.args.distributed:
                model = model.module
            torch.save(model.state_dict(), path)
            # update list and prune
            self.topk_checkpoints.append((score, path))
            self.topk_checkpoints.sort(key=lambda x: x[0], reverse=True)
            if len(self.topk_checkpoints) > self.topk_k:
                drop_score, drop_path = self.topk_checkpoints.pop()
                try:
                    if os.path.exists(drop_path):
                        os.remove(drop_path)
                except Exception:
                    pass
        except Exception as e:
            logger.info(f"Top-K checkpoint save skipped: {e}")

    def save_temperature_curve(self):
        if not self.temp_history:
            return
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            logger.info(f"Skipping temperature curve plot (matplotlib unavailable): {e}")
            return
        temps = torch.tensor(self.temp_history)
        iters = list(range(1, temps.size(0)+1))
        plt.figure()
        plt.plot(iters, temps[:,0], label='mean T')
        plt.fill_between(iters, temps[:,1], temps[:,2], color='orange', alpha=0.3, label='min-max band')
        plt.xlabel('Iteration')
        plt.ylabel('Temperature')
        plt.title('Dynamic Temperature over Training')
        plt.legend()
        out_path = os.path.join(self.args.save_dir, 'temperature_curve.png')
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved temperature curve to {out_path}")

        # Epoch-level diagnostic (mean T per epoch chunk). Epoch here is approximated by val_per_iters.
        epoch_size = max(self.args.val_per_iters, 1)
        epoch_means = []
        t_mean_series = temps[:,0]
        for i in range(0, t_mean_series.numel(), epoch_size):
            chunk = t_mean_series[i:i+epoch_size]
            epoch_means.append(chunk.mean().item())
        if epoch_means:
            plt.figure()
            plt.plot(range(1, len(epoch_means)+1), epoch_means, marker='o')
            plt.xlabel('Epoch (chunked by val_per_iters)')
            plt.ylabel('Temperature (mean)')
            plt.title('Epoch-wise Temperature (mean of temp_map)')
            out_path_epoch = os.path.join(self.args.save_dir, 'temperature_curve_epoch.png')
            plt.savefig(out_path_epoch, dpi=200, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved epoch-wise temperature curve to {out_path_epoch}")


def save_checkpoint(model, args, is_best=False):
    """Save Checkpoint"""
    directory = os.path.expanduser(args.save_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = 'kd_{}_{}_{}.pth'.format(args.student_model, args.student_backbone, args.dataset)
    filename = os.path.join(directory, filename)

    if args.distributed:
        model = model.module
    
    torch.save(model.state_dict(), filename)
    if is_best:
        best_filename = 'kd_{}_{}_{}_best_model.pth'.format(args.student_model, args.student_backbone, args.dataset)
        best_filename = os.path.join(directory, best_filename)
        shutil.copyfile(filename, best_filename)


if __name__ == '__main__':
    args = parse_args()

    # reference maskrcnn-benchmark
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.num_gpus = num_gpus
    args.distributed = num_gpus > 1
    if not args.no_cuda and torch.cuda.is_available():
        cudnn.benchmark = False
        args.device = "cuda"
    else:
        args.distributed = False
        args.device = "cpu"
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
        synchronize()

    logger = setup_logger("semantic_segmentation", args.save_dir, get_rank(), filename='{}_{}_{}_log.txt'.format(
        args.student_model, args.teacher_backbone, args.student_backbone, args.dataset))
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    trainer = Trainer(args)
    trainer.train()
    torch.cuda.empty_cache()
