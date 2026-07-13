import argparse
import time
import datetime
import os
import shutil
import math
import sys
import random
import numpy as np

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F

try:
    import torch_npu  # noqa: F401
    HAS_TORCH_NPU = True
except Exception:
    HAS_TORCH_NPU = False

from PCOS import get_max_confidence_and_residual_variance

from losses import *
from models.model_zoo import get_segmentation_model

from utils.sagan import Discriminator
from utils.distributed import *
from utils.logger import setup_logger
from utils.score import SegmentationMetric
from utils.flops import cal_multi_adds, cal_param_size
from utils.covar_temperature import (
    NewtonCoVarConfig,
    covar_temperature_kd_loss,
    newton_covar_temperature_map,
)
from utils.rtc_temperature import (
    RTCConfig,
    build_rtc_temperature_map,
    collect_rtc_diagnostics,
    file_sha256,
    load_frozen_reliability_cdf,
    masked_temperature_kd_loss,
)

# Lazy import datasets inside Trainer to avoid unnecessary deps


def npu_is_available():
    return HAS_TORCH_NPU and hasattr(torch, "npu") and torch.npu.is_available()


def resolve_device_type(requested):
    requested = str(requested).lower()
    if requested != 'auto':
        if requested == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('Requested --device-type cuda, but CUDA is not available.')
        if requested == 'npu' and not npu_is_available():
            raise RuntimeError('Requested --device-type npu, but torch_npu/NPU is not available.')
        return requested
    if torch.cuda.is_available():
        return 'cuda'
    if npu_is_available():
        return 'npu'
    return 'cpu'


def set_accelerator_device(device_type, local_rank):
    if device_type == 'cuda':
        torch.cuda.set_device(local_rank)
    elif device_type == 'npu':
        torch.npu.set_device(local_rank)


def empty_accelerator_cache(device):
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'npu':
        torch.npu.empty_cache()


def seed_everything(seed, rank=0):
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if npu_is_available():
        torch.npu.manual_seed_all(seed)


def unwrap_module(module):
    return module.module if isinstance(module, nn.parallel.DistributedDataParallel) else module


def load_state_dict_compatible(module, state_dict, strict=True):
    """Load state dicts saved with or without a DistributedDataParallel prefix."""
    if state_dict is None:
        return
    cleaned = {
        key[7:] if key.startswith('module.') else key: value
        for key, value in state_dict.items()
    }
    unwrap_module(module).load_state_dict(cleaned, strict=strict)


def _move_value_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_value_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_value_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_value_to_device(item, device) for item in value)
    return value


def move_optimizer_state_to_device(optimizer, device):
    if optimizer is None:
        return
    for state_id, state in optimizer.state.items():
        optimizer.state[state_id] = _move_value_to_device(state, device)


def capture_rng_state():
    """Capture RNGs that can affect training after a checkpoint boundary."""
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state()
    if npu_is_available() and hasattr(torch.npu, 'get_rng_state'):
        state['npu'] = torch.npu.get_rng_state()
    return state


def restore_rng_state(state):
    """Restore a state produced by capture_rng_state."""
    if not state:
        return
    if 'python' in state:
        random.setstate(state['python'])
    if 'numpy' in state:
        np.random.set_state(state['numpy'])
    if 'torch' in state:
        torch.set_rng_state(state['torch'].cpu())
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state(state['cuda'])
    if (
        'npu' in state
        and npu_is_available()
        and hasattr(torch.npu, 'set_rng_state')
    ):
        torch.npu.set_rng_state(state['npu'])


def load_checkpoint_file(path):
    """Load trusted local checkpoints across torch versions."""
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:  # torch < 2.0 has no weights_only argument
        return torch.load(path, map_location='cpu')


def is_training_state_checkpoint(checkpoint):
    if not isinstance(checkpoint, dict):
        return False
    return (
        checkpoint.get('checkpoint_type') == 'train_kd_training_state'
        or (
            ('student' in checkpoint or 'state_dict' in checkpoint)
            and any(key in checkpoint for key in ('optimizer', 'iteration', 'rng_state'))
        )
    )


def extract_student_state_dict(checkpoint):
    if is_training_state_checkpoint(checkpoint):
        return checkpoint.get('student', checkpoint.get('state_dict'))
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        return checkpoint['state_dict']
    if isinstance(checkpoint, dict) and 'student' in checkpoint:
        return checkpoint['student']
    return checkpoint


def format_sample_validation_log(sample, pix_acc, miou):
    return "Sample: {:d}, Validation pixAcc: {:.6f}, mIoU: {:.6f}".format(
        int(sample), float(pix_acc), float(miou)
    )


def format_overall_validation_log(pix_acc, miou):
    return "Overall validation pixAcc: {:.6f}, mIoU: {:.6f}".format(
        float(pix_acc), float(miou)
    )

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


    parser.add_argument("--kd-temperature", type=float, default=1.0, help="logits KD temperature")
    parser.add_argument('--kd-loss-mode', type=str, default='legacy',
                        choices=['legacy', 'masked'],
                        help='legacy unmasked scalar KD or fair masked pixel KD')
    parser.add_argument("--lambda-kd", type=float, default=0., help="lambda_kd")
    parser.add_argument("--lambda-adv", type=float, default=0., help="lambda adversarial loss")
    parser.add_argument("--lambda-d", type=float, default=0., help="lambda discriminator loss")
    parser.add_argument("--lambda-skd", type=float, default=0., help="lambda skd")
    parser.add_argument("--lambda-cwd-fea", type=float, default=0., help="lambda cwd feature")
    parser.add_argument("--lambda-cwd-logit", type=float, default=0., help="lambda cwd logit")
    parser.add_argument("--lambda-ifv", type=float, default=0., help="lambda ifvd")
    parser.add_argument("--lambda-fitnet", type=float, default=0., help="lambda fitnet")
    parser.add_argument("--lambda-at", type=float, default=0., help="lambda attention transfer")
    parser.add_argument("--lambda-psd", type=float, default=0., help="lambda pixel similarity KD")
    parser.add_argument("--lambda-csd", type=float, default=0., help="lambda category similarity KD")
    parser.add_argument('--use-covar', action='store_true', default=False,
                        help='enable CoVar in the logit KD path')
    parser.add_argument('--covar-temp-mode', type=str, default='newton',
                        choices=['newton', 'legacy_weight', 'rtc'],
                        help='paper Newton temperature or the earlier Gaussian weighting implementation')
    parser.add_argument('--covar-alpha', type=float, default=2.0,
                        help='alpha used only by legacy_weight mode')
    parser.add_argument('--teacher-output-temp', type=float, default=1.0,
                        help='soften teacher logits in the logit KD path only')
    parser.add_argument('--covar-temp-base', type=float, default=1.0,
                        help='initial temperature for Newton CoVar')
    parser.add_argument('--covar-temp-min', type=float, default=0.5,
                        help='minimum pixel temperature for Newton CoVar')
    parser.add_argument('--covar-temp-max', type=float, default=8.0,
                        help='maximum pixel temperature for Newton CoVar')
    parser.add_argument('--covar-kd-temp-power', type=float, default=2.0,
                        help='gamma in KL(student/T, teacher/T) * T^gamma')
    parser.add_argument('--covar-grad-eta', type=float, default=0.6,
                        help='damping factor for Newton temperature updates')
    parser.add_argument('--covar-grad-max-iter', type=int, default=8,
                        help='number of Newton temperature updates')
    parser.add_argument('--covar-a', type=float, default=None,
                        help='reliability variance coefficient; default is (K-1)^2/2')
    parser.add_argument('--covar-reliability-mode', type=str, default='full',
                        choices=['full', 'confidence', 'variance'],
                        help='reliability terms used by Newton CoVar')
    parser.add_argument('--covar-newton-hessian-eps', type=float, default=1e-5,
                        help='minimum positive Hessian magnitude for a Newton step')
    parser.add_argument('--covar-newton-max-step', type=float, default=0.25,
                        help='maximum absolute Newton step; <=0 disables clipping')
    parser.add_argument('--rtc-cdf-path', type=str, default=None,
                        help='frozen training-set reliability CDF artifact')
    parser.add_argument('--rtc-assess-temperature', type=float, default=1.0,
                        help='fixed temperature used only for raw-teacher reliability')
    parser.add_argument('--rtc-route-quantile', type=float, default=0.80,
                        help='high-risk route starts above this frozen-CDF quantile')
    parser.add_argument('--rtc-route-width', type=float, default=0.05,
                        help='continuous route transition width')
    parser.add_argument('--rtc-temp-reliable', type=float, default=0.5,
                        help='strong-sharpening endpoint temperature')
    parser.add_argument('--rtc-temp-neutral', type=float, default=1.0,
                        help='neutral pixel temperature')
    parser.add_argument('--rtc-temp-unreliable', type=float, default=2.0,
                        help='smoothing endpoint temperature')
    parser.add_argument('--rtc-alpha-reliable', type=float, default=1.0,
                        help='fraction of the reliable endpoint target')
    parser.add_argument('--rtc-alpha-unreliable', type=float, default=1.0,
                        help='fraction of the unreliable endpoint target')
    parser.add_argument(
        '--rtc-enable-reliable',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='explicitly enable/disable the low-risk sharpening branch',
    )
    parser.add_argument(
        '--rtc-enable-unreliable',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='explicitly enable/disable the high-risk smoothing branch',
    )
    parser.add_argument('--rtc-bisection-iters', type=int, default=16,
                        help='fixed target-temperature bisection iterations')
    parser.add_argument('--rtc-shuffle', action=argparse.BooleanOptionalAction, default=None,
                        help='explicitly enable/disable within-image temperature shuffling')
    parser.add_argument('--rtc-reverse-routing', action=argparse.BooleanOptionalAction, default=None,
                        help='explicitly enable/disable swapping reliable and unreliable gates')
               

    # accelerator setting
    parser.add_argument('--device-type', type=str, default='auto', choices=['auto', 'cuda', 'npu', 'cpu'],
                        help='accelerator backend; auto prefers CUDA, then Ascend NPU, then CPU')
    parser.add_argument('--seed', type=int, default=1234,
                        help='base random seed for reproducible training')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--local-rank', type=int, default=0)
    # checkpoint and log
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--save-dir', default='~/.torch/models',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--save-epoch', type=int, default=10,
                        help='save model every checkpoint-epoch')
    parser.add_argument('--log-dir', default='../runs/logs/',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--log-iter', type=int, default=10,
                        help='print log every log-iter')
    parser.add_argument('--save-per-iters', type=int, default=800,
                        help='per iters to save')
    parser.add_argument('--val-per-iters', type=int, default=800,
                        help='per iters to val')
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
    if args.teacher_output_temp <= 0:
        parser.error('--teacher-output-temp must be positive')
    if args.kd_temperature <= 0:
        parser.error('--kd-temperature must be positive')
    if args.use_covar and args.covar_temp_mode == 'rtc':
        explicit_rtc_flags = {
            '--rtc-enable-reliable': args.rtc_enable_reliable,
            '--rtc-enable-unreliable': args.rtc_enable_unreliable,
            '--rtc-shuffle': args.rtc_shuffle,
            '--rtc-reverse-routing': args.rtc_reverse_routing,
        }
        missing_flags = [
            name for name, value in explicit_rtc_flags.items() if value is None
        ]
        if missing_flags:
            parser.error(
                'RTC requires explicit Boolean choices: {}'.format(
                    ', '.join(missing_flags)
                )
            )
        if args.kd_loss_mode != 'masked':
            parser.error('RTC requires --kd-loss-mode masked')
        if not args.rtc_cdf_path:
            parser.error('RTC requires --rtc-cdf-path')
        if args.dataset != 'voc':
            parser.error('Phase O RTC currently supports only VOC with a matching frozen CDF')
        if args.resume:
            parser.error('Phase O RTC formal runs forbid --resume; start a fresh run directory')
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    if num_gpus > 1 and args.local_rank == 0:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    if args.student_backbone.startswith('resnet'):
        args.aux = True
    elif args.student_backbone.startswith('mobile'):
        args.aux = False
    else:
        raise ValueError('no such network')

    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args
        if args.distributed and args.device in ("cuda", "npu"):
            self.device = torch.device(f"{args.device}:{args.local_rank}")
        else:
            self.device = torch.device(args.device)
        self.num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
        self.rank = get_rank()
        self.resume_checkpoint = None
        self.resume_is_full_state = False
        self.start_iteration = 0
        if args.resume:
            if not os.path.isfile(args.resume):
                raise FileNotFoundError('Resume checkpoint not found: {}'.format(args.resume))
            extension = os.path.splitext(args.resume)[1].lower()
            if extension not in ('.pth', '.pkl'):
                raise ValueError('Only .pth and .pkl checkpoints are supported.')
            print('Resuming training, loading {}...'.format(args.resume))
            self.resume_checkpoint = load_checkpoint_file(args.resume)
            self.resume_is_full_state = is_training_state_checkpoint(self.resume_checkpoint)
            if self.resume_is_full_state:
                self.start_iteration = int(self.resume_checkpoint.get('iteration', 0))
                if self.start_iteration > args.max_iterations:
                    raise ValueError(
                        'Checkpoint iteration {} exceeds --max-iterations {}'.format(
                            self.start_iteration, args.max_iterations
                        )
                    )

        self.covar_newton_config = None
        if args.use_covar and args.covar_temp_mode == 'newton':
            self.covar_newton_config = NewtonCoVarConfig(
                base_temperature=args.covar_temp_base,
                min_temperature=args.covar_temp_min,
                max_temperature=args.covar_temp_max,
                kd_temperature_power=args.covar_kd_temp_power,
                eta=args.covar_grad_eta,
                max_iterations=args.covar_grad_max_iter,
                hessian_epsilon=args.covar_newton_hessian_eps,
                max_step=args.covar_newton_max_step,
                coefficient_a=args.covar_a,
                reliability_mode=args.covar_reliability_mode,
            )
            self.covar_newton_config.validate()

        self.rtc_config = None
        self.rtc_cdf = None
        if args.use_covar and args.covar_temp_mode == 'rtc':
            self.rtc_config = RTCConfig(
                assess_temperature=args.rtc_assess_temperature,
                route_quantile=args.rtc_route_quantile,
                route_width=args.rtc_route_width,
                reliable_temperature=args.rtc_temp_reliable,
                neutral_temperature=args.rtc_temp_neutral,
                unreliable_temperature=args.rtc_temp_unreliable,
                alpha_reliable=args.rtc_alpha_reliable,
                alpha_unreliable=args.rtc_alpha_unreliable,
                enable_reliable=args.rtc_enable_reliable,
                enable_unreliable=args.rtc_enable_unreliable,
                bisection_iterations=args.rtc_bisection_iters,
                kd_temperature_power=args.covar_kd_temp_power,
                coefficient_a=args.covar_a,
                reliability_mode=args.covar_reliability_mode,
            )
            self.rtc_config.validate()


        # Lazy dataset imports to avoid importing unused backends
        if args.dataset == 'citys':
            from dataset.cityscapes import CSTrainValSet
            train_dataset = CSTrainValSet(args.data, 
                                            list_path='./dataset/list/cityscapes/train.lst', 
                                            max_iters=args.max_iterations*args.batch_size, 
                                            crop_size=args.crop_size, scale=True, mirror=True)
            val_dataset = CSTrainValSet(args.data, 
                                        list_path='./dataset/list/cityscapes/val.lst', 
                                        crop_size=(1024, 2048), scale=False, mirror=False)
        elif args.dataset == 'voc':
            from dataset.voc import VOCDataTrainSet, VOCDataValSet
            self._assert_voc_aug_paths(args.data)
            if not args.skip_val:
                self._assert_voc_val_paths(args.data)
            train_dataset = VOCDataTrainSet(args.data, './dataset/list/voc/train_aug.txt', max_iters=args.max_iterations*args.batch_size, 
                                          crop_size=args.crop_size, scale=True, mirror=True, ignore_label=args.ignore_label)
            val_dataset = VOCDataValSet(args.data, './dataset/list/voc/val.txt', ignore_label=args.ignore_label)
        elif args.dataset == 'ade20k':
            from dataset.ade20k import ADETrainSet, ADEDataValSet
            train_dataset = ADETrainSet(args.data, max_iters=args.max_iterations*args.batch_size, ignore_label=args.ignore_label,
                                        crop_size=args.crop_size, scale=True, mirror=True)
            val_dataset = ADEDataValSet(args.data)
        elif args.dataset == 'camvid':
            from dataset.camvid import CamvidTrainSet, CamvidValSet
            train_dataset = CamvidTrainSet(args.data, './dataset/list/CamVid/camvid_train_list.txt', max_iters=args.max_iterations*args.batch_size,
                            ignore_label=args.ignore_label, crop_size=args.crop_size, scale=True, mirror=True)
            val_dataset = CamvidValSet(args.data, './dataset/list/CamVid/camvid_val_list.txt')
        elif args.dataset == 'coco_stuff_164k':
            from dataset.coco_stuff_164k import CocoStuff164kTrainSet, CocoStuff164kValSet
            train_dataset = CocoStuff164kTrainSet(args.data, './dataset/list/coco_stuff_164k/coco_stuff_164k_train.txt', max_iters=args.max_iterations*args.batch_size, ignore_label=args.ignore_label,
                                        crop_size=args.crop_size, scale=True, mirror=True)
            val_dataset = CocoStuff164kValSet(args.data, './dataset/list/coco_stuff_164k/coco_stuff_164k_val.txt')
        else:
            raise ValueError('dataset unfind')

        if self.rtc_config is not None:
            self.rtc_cdf = load_frozen_reliability_cdf(
                args.rtc_cdf_path,
                device=self.device,
            )
            self._validate_rtc_cdf_metadata(self.rtc_cdf, train_dataset.num_class)
            cdf_checksums = all_gather(self.rtc_cdf.checksum_sha256)
            if len(set(cdf_checksums)) != 1:
                raise RuntimeError(
                    'RTC CDF checksum differs across ranks: {}'.format(cdf_checksums)
                )
            args.rtc_cdf_path = self.rtc_cdf.path
            args.rtc_cdf_sha256 = self.rtc_cdf.checksum_sha256
            logger.info(
                'Loaded frozen RTC CDF: path={} sha256={} metadata={}'.format(
                    self.rtc_cdf.path, self.rtc_cdf.checksum_sha256,
                    dict(self.rtc_cdf.metadata),
                )
            )

    
        args.batch_size = args.batch_size // self.num_gpus
        train_sampler = make_data_sampler(train_dataset, shuffle=True, distributed=args.distributed)
        train_batch_sampler = make_batch_data_sampler(
            train_sampler,
            args.batch_size,
            args.max_iterations,
            start_iter=self.start_iteration,
        )
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

        self.use_adv = args.lambda_adv != 0. or args.lambda_d != 0.
        self.D_model = None
        if self.use_adv:
            self.D_model = Discriminator(
                preprocess_GAN_mode=1,
                input_channel=train_dataset.num_class,
                distributed=args.distributed,
            ).to(self.device)

        # create criterion
        x = torch.randn(1, 3, args.crop_size[0], args.crop_size[1]).to(self.device)
        t_y = self.t_model(x)
        s_y = self.s_model(x)
        t_channels = t_y[-1].size(1)
        s_channels = s_y[-1].size(1)

        self.criterion = SegCrossEntropyLoss(ignore_index=args.ignore_label).to(self.device)
        self.criterion_kd = CriterionKD(temperature=args.kd_temperature).to(self.device)
        self.criterion_adv = CriterionAdv('hinge').to(self.device)
        self.criterion_adv_for_G = CriterionAdvForG('hinge').to(self.device)
        self.criterion_skd = CriterionStructuralKD().to(self.device)
        self.criterion_ifv = CriterionIFV(train_dataset.num_class).to(self.device)
        self.criterion_cwd = CriterionCWD(s_channels, t_channels, norm_type='channel',divergence='kl', temperature=4.).to(self.device)
        self.criterion_fitnet = CriterionFitNet(s_channels, t_channels).to(self.device)
        self.criterion_at = CriterionAT().to(self.device)
        self.criterion_dsd = CriterionDoubleSimKD().to(self.device)

    
        params_list = nn.ModuleList([])
        params_list.append(self.s_model)
        params_list.append(self.criterion_cwd)
        params_list.append(self.criterion_fitnet)


        self.optimizer = torch.optim.SGD(params_list.parameters(),
                                         lr=args.lr,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay)

        self.D_optimizer = None
        if self.use_adv:
            self.D_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                                self.D_model.parameters()),
                                                4e-4, [0.9, 0.99])
        
        if args.distributed:
            ddp_kwargs = {}
            if self.device.type in ('cuda', 'npu'):
                ddp_kwargs = {'device_ids': [args.local_rank], 'output_device': args.local_rank}
            self.s_model = nn.parallel.DistributedDataParallel(self.s_model, **ddp_kwargs)
            if self.use_adv:
                self.D_model = nn.parallel.DistributedDataParallel(self.D_model, **ddp_kwargs)
            self.criterion_cwd = nn.parallel.DistributedDataParallel(self.criterion_cwd, **ddp_kwargs)
            self.criterion_fitnet = nn.parallel.DistributedDataParallel(self.criterion_fitnet, **ddp_kwargs)
            
        # evaluation metrics
        self.metric = SegmentationMetric(train_dataset.num_class)
        self.best_pred = 0.0
        self.current_iteration = self.start_iteration
        if self.resume_checkpoint is not None:
            self._load_resume_checkpoint(self.resume_checkpoint, args.resume)

    def _validate_rtc_cdf_metadata(self, cdf, num_classes):
        metadata = dict(cdf.metadata)
        required = (
            'num_classes',
            'assess_temperature',
            'coefficient_a',
            'reliability_mode',
            'teacher_sha256',
            'train_list_sha256',
            'dataset',
            'split',
            'processed_images',
            'dataset_size',
            'max_images',
            'full_dataset_scan',
            'batch_size',
            'workers',
            'valid_native_pixels',
            'finite_valid_pixels',
            'nonfinite_valid_pixels',
            'sample_count',
            'num_quantiles',
            'max_pixels_per_image',
            'crop_size',
            'scale',
            'mirror',
            'seed',
            'teacher_output_grid',
            'valid_mask_resize',
            'source_sha256',
        )
        missing = [key for key in required if key not in metadata]
        if missing:
            raise ValueError('RTC CDF metadata missing fields: {}'.format(missing))
        if metadata['dataset'] != 'voc':
            raise ValueError('Phase O RTC requires a VOC training-set CDF')
        if int(metadata['processed_images']) != int(metadata['dataset_size']):
            raise ValueError('RTC CDF was not built from a complete dataset scan')
        if int(metadata['num_classes']) != int(num_classes):
            raise ValueError('RTC CDF num_classes does not match the dataset')
        pre_registered_metadata = {
            'split': 'train_aug',
            'max_images': 0,
            'full_dataset_scan': True,
            'batch_size': 4,
            'workers': 0,
            'max_pixels_per_image': 4096,
            'num_quantiles': 4097,
            'crop_size': [512, 512],
            'scale': True,
            'mirror': True,
            'seed': 1234,
            'teacher_output_grid': 'native',
            'valid_mask_resize': 'nearest',
        }
        for key, expected_value in pre_registered_metadata.items():
            if metadata[key] != expected_value:
                raise ValueError(
                    'RTC CDF metadata drift for {}: expected={} observed={}'.format(
                        key, expected_value, metadata[key]
                    )
                )
        for boolean_key in ('full_dataset_scan', 'scale', 'mirror'):
            if metadata[boolean_key] is not True:
                raise ValueError(
                    'RTC CDF metadata {} must be the Boolean true'.format(
                        boolean_key
                    )
                )
        processed_images = int(metadata['processed_images'])
        dataset_size = int(metadata['dataset_size'])
        valid_native_pixels = int(metadata['valid_native_pixels'])
        finite_valid_pixels = int(metadata['finite_valid_pixels'])
        nonfinite_valid_pixels = int(metadata['nonfinite_valid_pixels'])
        sampled_pixels = int(metadata['sample_count'])
        if processed_images <= 0 or dataset_size <= 0:
            raise ValueError('RTC CDF dataset counters must be positive')
        if valid_native_pixels != finite_valid_pixels + nonfinite_valid_pixels:
            raise ValueError('RTC CDF valid-pixel counters are inconsistent')
        if nonfinite_valid_pixels != 0:
            raise ValueError('RTC CDF formal scan contains non-finite valid pixels')
        if sampled_pixels <= 0 or sampled_pixels > finite_valid_pixels:
            raise ValueError('RTC CDF sampled-pixel count is invalid')
        source_sha256 = metadata['source_sha256']
        if not isinstance(source_sha256, dict):
            raise ValueError('RTC CDF source_sha256 must be a dictionary')
        expected_rtc_sha256 = file_sha256(
            os.path.join(cur_path, 'utils', 'rtc_temperature.py')
        )
        if source_sha256.get('rtc_temperature') != expected_rtc_sha256:
            raise ValueError(
                'RTC CDF was built with a different RTC reliability implementation'
            )
        if not math.isclose(
            float(metadata['assess_temperature']),
            float(self.rtc_config.assess_temperature),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError('RTC CDF assess_temperature does not match the run')
        expected_a = self.rtc_config.coefficient_a
        if expected_a is None:
            expected_a = float((int(num_classes) - 1) ** 2) / 2.0
        if not math.isclose(
            float(metadata['coefficient_a']), float(expected_a),
            rel_tol=0.0, abs_tol=1e-12,
        ):
            raise ValueError('RTC CDF coefficient_a does not match the run')
        if metadata['reliability_mode'] != self.rtc_config.reliability_mode:
            raise ValueError('RTC CDF reliability_mode does not match the run')
        if file_sha256(self.args.teacher_pretrained) != metadata['teacher_sha256']:
            raise ValueError('RTC CDF teacher checksum does not match the run')
        train_list_path = './dataset/list/voc/train_aug.txt'
        if self.args.dataset == 'voc' and file_sha256(train_list_path) != metadata['train_list_sha256']:
            raise ValueError('RTC CDF train-list checksum does not match the run')

    def _load_resume_checkpoint(self, checkpoint, path):
        if self.rtc_config is not None:
            saved_rtc = checkpoint.get('rtc') if isinstance(checkpoint, dict) else None
            if not saved_rtc:
                raise ValueError('RTC resume checkpoint has no frozen RTC metadata')
            expected = self._rtc_checkpoint_metadata()
            for key in (
                'config', 'cdf_sha256', 'kd_loss_mode',
                'teacher_output_temp', 'shuffle', 'reverse_routing', 'world_size',
            ):
                if saved_rtc.get(key) != expected.get(key):
                    raise ValueError(
                        'RTC resume configuration drift for {}: saved={} current={}'.format(
                            key, saved_rtc.get(key), expected.get(key)
                        )
                    )
        student_state = extract_student_state_dict(checkpoint)
        load_state_dict_compatible(self.s_model, student_state, strict=True)
        if not self.resume_is_full_state:
            logger.info(
                'Loaded legacy student-only checkpoint from {}; training starts at iteration 0'.format(path)
            )
            return

        module_states = (
            ('criterion_cwd', self.criterion_cwd),
            ('criterion_fitnet', self.criterion_fitnet),
            ('D', self.D_model),
        )
        for key, module in module_states:
            if module is not None and checkpoint.get(key) is not None:
                load_state_dict_compatible(module, checkpoint[key], strict=True)

        if checkpoint.get('optimizer') is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            move_optimizer_state_to_device(self.optimizer, self.device)
        if self.D_optimizer is not None and checkpoint.get('D_optimizer') is not None:
            self.D_optimizer.load_state_dict(checkpoint['D_optimizer'])
            move_optimizer_state_to_device(self.D_optimizer, self.device)

        self.start_iteration = int(checkpoint.get('iteration', self.start_iteration))
        self.current_iteration = self.start_iteration
        self.best_pred = float(checkpoint.get('best_pred', self.best_pred))
        rng_state = checkpoint.get('rng_state')
        rank_states = checkpoint.get('rng_state_by_rank')
        if isinstance(rank_states, (list, tuple)) and self.rank < len(rank_states):
            rng_state = rank_states[self.rank]
        elif isinstance(rank_states, dict):
            rng_state = rank_states.get(str(self.rank), rank_states.get(self.rank, rng_state))
        restore_rng_state(rng_state)
        logger.info(
            'Resumed full training state from {}: iteration={}, best_mIoU={:.6f}'.format(
                path, self.start_iteration, self.best_pred
            )
        )

    def _rtc_checkpoint_metadata(self):
        if self.rtc_config is None:
            return None
        return {
            'config': self.rtc_config.to_dict(),
            'cdf_path': self.rtc_cdf.path,
            'cdf_sha256': self.rtc_cdf.checksum_sha256,
            'cdf_metadata': dict(self.rtc_cdf.metadata),
            'kd_loss_mode': self.args.kd_loss_mode,
            'teacher_output_temp': float(self.args.teacher_output_temp),
            'shuffle': bool(self.args.rtc_shuffle),
            'reverse_routing': bool(self.args.rtc_reverse_routing),
            'world_size': int(get_world_size()),
        }

    def _training_state_dict(self, iteration, rng_states):
        return {
            'checkpoint_type': 'train_kd_training_state',
            'checkpoint_version': 3,
            'student': unwrap_module(self.s_model).state_dict(),
            'criterion_cwd': unwrap_module(self.criterion_cwd).state_dict(),
            'criterion_fitnet': unwrap_module(self.criterion_fitnet).state_dict(),
            'D': None if self.D_model is None else unwrap_module(self.D_model).state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'D_optimizer': None if self.D_optimizer is None else self.D_optimizer.state_dict(),
            'iteration': int(iteration),
            'best_pred': float(self.best_pred),
            'rng_state': rng_states[0],
            'rng_state_by_rank': rng_states,
            'world_size': int(get_world_size()),
            'args': dict(vars(self.args)),
            'rtc': self._rtc_checkpoint_metadata(),
        }

    def save_checkpoint(self, is_best=False, iteration=None, save_latest=True):
        iteration = self.current_iteration if iteration is None else int(iteration)
        local_rng_state = capture_rng_state()
        rng_states = all_gather(local_rng_state)
        if get_rank() != 0:
            return
        training_state = self._training_state_dict(iteration, rng_states)
        save_checkpoint(
            self.s_model,
            self.args,
            is_best=is_best,
            training_state=training_state,
            save_latest=save_latest,
        )

    @staticmethod
    def _assert_voc_aug_paths(root):
        jpeg_dir = os.path.join(root, 'JPEGImages')
        aug_dir = os.path.join(root, 'SegmentationClassAug')
        missing = []
        if not os.path.isdir(jpeg_dir):
            missing.append(jpeg_dir)
        if not os.path.isdir(aug_dir):
            missing.append(aug_dir)
        if missing:
            raise ValueError(f"VOC data root invalid: missing directories: {', '.join(missing)}. "
                             f"Please set --data to VOCAug root with JPEGImages/ and SegmentationClassAug/.")

    @staticmethod
    def _assert_voc_val_paths(root):
        jpeg_dir = os.path.join(root, 'JPEGImages')
        val_dir = os.path.join(root, 'SegmentationClass')
        missing = []
        if not os.path.isdir(jpeg_dir):
            missing.append(jpeg_dir)
        if not os.path.isdir(val_dir):
            missing.append(val_dir)
        if missing:
            raise ValueError(f"VOC validation requires: {', '.join(missing)}. "
                             f"If using VOCAug root, pass --skip-val or set --data to VOC2012 root for validation.")

    def adjust_lr(self, base_lr, iter, max_iter, power):
        cur_lr = base_lr*((1-float(iter)/max_iter)**(power))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = cur_lr

        return cur_lr

    def reduce_tensor(self, tensor):
        if not self.args.distributed:
            return tensor
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        return rt

    def reduce_mean_tensor(self, tensor):
        if not self.args.distributed:
            return tensor
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.num_gpus
        return rt

    def aggregate_rtc_diagnostics(self, maps):
        """Compute exact global batch diagnostics instead of averaging rank quantiles."""
        valid = maps.valid_mask
        active = (
            ((maps.gate_reliable > 0) & (self.rtc_config.alpha_reliable > 0))
            | ((maps.gate_unreliable > 0) & (self.rtc_config.alpha_unreliable > 0))
        )
        active = (
            valid
            & active
            & ~maps.fallback_mask
            & ~maps.tie_mask
            & maps.finite_mask
        )
        local_payload = {
            'temperature': maps.temperature[valid].detach().float().cpu(),
            'reliability': maps.reliability[valid].detach().float().cpu(),
            'quantile': maps.reliability_quantile[valid].detach().float().cpu(),
            'gate_reliable': maps.gate_reliable[valid].detach().float().cpu(),
            'gate_unreliable': maps.gate_unreliable[valid].detach().float().cpu(),
            'residual_active': maps.target_residual[active].detach().float().cpu(),
            'fallback': maps.fallback_mask[valid].detach().float().cpu(),
            'tie': maps.tie_mask[valid].detach().float().cpu(),
            'finite': maps.finite_mask[valid].detach().float().cpu(),
            'shuffled': bool(maps.shuffled),
        }
        gathered = all_gather(local_payload)

        def concatenate(key):
            values = [item[key] for item in gathered if item[key].numel() > 0]
            return torch.cat(values) if values else torch.empty(0, dtype=torch.float32)

        temperature = concatenate('temperature')
        if temperature.numel() == 0:
            diagnostics = collect_rtc_diagnostics(
                maps,
                self.rtc_config,
                teacher_output_temperature=self.args.teacher_output_temp,
            )
            diagnostics['target_residual_active_count'] = 0.0
            return diagnostics
        reliability = concatenate('reliability')
        quantile = concatenate('quantile')
        gate_reliable = concatenate('gate_reliable')
        gate_unreliable = concatenate('gate_unreliable')
        residual = concatenate('residual_active')
        fallback = concatenate('fallback')
        tie = concatenate('tie')
        finite = concatenate('finite')
        shuffled = any(item['shuffled'] for item in gathered)
        temperature_quantiles = torch.quantile(
            temperature, torch.tensor([0.10, 0.50, 0.90, 0.95])
        )
        bisection_scale = math.ldexp(
            1.0, -(self.rtc_config.bisection_iterations + 1)
        )
        floating_tolerance = (
            4.0
            * torch.finfo(temperature.dtype).eps
            * max(1.0, self.rtc_config.unreliable_temperature)
        )
        reliable_tolerance = (
            (self.rtc_config.neutral_temperature
             - self.rtc_config.reliable_temperature)
            * bisection_scale
            + floating_tolerance
        )
        unreliable_tolerance = (
            (self.rtc_config.unreliable_temperature
             - self.rtc_config.neutral_temperature)
            * bisection_scale
            + floating_tolerance
        )
        neutral_tolerance = max(
            reliable_tolerance, unreliable_tolerance
        )
        residual_is_applicable = not shuffled
        if residual.numel() > 0 and residual_is_applicable:
            residual_p95 = float(torch.quantile(residual, 0.95))
            residual_mean = float(residual.mean())
            residual_max = float(residual.max())
        else:
            residual_p95 = 0.0
            residual_mean = 0.0
            residual_max = 0.0
        return {
            'valid_count': float(temperature.numel()),
            'reliability_mean': float(reliability.mean()),
            'quantile_mean': float(quantile.mean()),
            'reliable_coverage': float((gate_reliable > 0).float().mean()),
            'unreliable_coverage': float((gate_unreliable > 0).float().mean()),
            'reliable_gate_mean': float(gate_reliable.mean()),
            'unreliable_gate_mean': float(gate_unreliable.mean()),
            'temperature_mean': float(temperature.mean()),
            'temperature_harmonic_mean': float(1.0 / (1.0 / temperature).mean()),
            'temperature_q10': float(temperature_quantiles[0]),
            'temperature_q50': float(temperature_quantiles[1]),
            'temperature_q90': float(temperature_quantiles[2]),
            'temperature_p95': float(temperature_quantiles[3]),
            'temperature_reliable_endpoint_rate': float(
                (torch.abs(temperature - self.rtc_config.reliable_temperature)
                 <= reliable_tolerance).float().mean()
            ),
            'temperature_neutral_rate': float(
                (torch.abs(temperature - self.rtc_config.neutral_temperature)
                 <= neutral_tolerance).float().mean()
            ),
            'temperature_unreliable_endpoint_rate': float(
                (torch.abs(temperature - self.rtc_config.unreliable_temperature)
                 <= unreliable_tolerance).float().mean()
            ),
            'effective_teacher_temperature_mean': float(
                temperature.mean() * self.args.teacher_output_temp
            ),
            'target_residual_active_count': float(residual.numel()),
            'target_residual_mean': residual_mean,
            'target_residual_p95': residual_p95,
            'target_residual_max': residual_max,
            'residual_is_applicable': float(residual_is_applicable),
            'fallback_rate': float(fallback.mean()),
            'tie_rate': float(tie.mean()),
            'finite_rate': float(finite.mean()),
            'shuffled': float(shuffled),
        }

    @staticmethod
    def batch_class_stats(max_confidence, residual_variance, valid_mask, epsilon=1e-8):
        valid = valid_mask.float()
        denom = valid.sum(dim=(1, 2)).clamp_min(1.0)

        conf_mean = (max_confidence * valid).sum(dim=(1, 2)) / denom
        res_mean = (residual_variance * valid).sum(dim=(1, 2)) / denom

        conf_var = ((max_confidence - conf_mean.view(-1, 1, 1)) ** 2 * valid).sum(dim=(1, 2)) / denom
        res_var = ((residual_variance - res_mean.view(-1, 1, 1)) ** 2 * valid).sum(dim=(1, 2)) / denom

        means = torch.stack((conf_mean, res_mean), dim=1)
        vars = torch.stack((conf_var, res_var), dim=1)
        return means, vars

    @torch.no_grad()
    def get_covar_weight(self, pred_prob, valid_mask, epsilon=1e-8):
        num_classes = pred_prob.size(1)
        max_confidence, residual_variance = get_max_confidence_and_residual_variance(
            pred_prob, valid_mask, num_classes, epsilon=epsilon)

        means, vars = self.batch_class_stats(max_confidence, residual_variance, valid_mask, epsilon)

        conf_mean = means[:, 0].view(-1, 1, 1)
        res_mean = means[:, 1].view(-1, 1, 1)
        conf_var = vars[:, 0].view(-1, 1, 1)
        res_var = vars[:, 1].view(-1, 1, 1)

        conf_z = (max_confidence - conf_mean) / torch.sqrt(conf_var + epsilon)
        res_z = (res_mean - residual_variance) / torch.sqrt(res_var + epsilon)

        weight_conf = torch.exp(- (conf_z ** 2) / self.args.covar_alpha)
        weight_res = torch.exp(- (res_z ** 2) / self.args.covar_alpha)

        weight = weight_conf * weight_res
        confident_mask = (conf_z > 0) | (res_z > 0)
        weight = torch.where(confident_mask, torch.ones_like(weight), weight)

        weight_mask = torch.where(valid_mask, weight, torch.zeros_like(weight))
        return weight_mask

    def covar_weighted_kd_loss(self, student_logits, teacher_logits, weight, epsilon=1e-8):
        temperature = self.args.kd_temperature
        s_log_prob = F.log_softmax(student_logits / temperature, dim=1)
        t_prob = F.softmax(teacher_logits / temperature, dim=1)

        kd_map = F.kl_div(s_log_prob, t_prob, reduction='none').sum(dim=1)
        weighted_loss = (kd_map * weight).sum() / weight.sum().clamp_min(epsilon)
        return weighted_loss * (temperature ** 2)

    def train(self):
        save_to_disk = get_rank() == 0
        log_per_iters, val_per_iters = self.args.log_iter, self.args.val_per_iters
        save_per_iters = self.args.save_per_iters
        start_time = time.time()
        logger.info('Start training, Total Iterations {:d}'.format(self.args.max_iterations))
        if self.start_iteration:
            logger.info('Continuing training from iteration {:d}'.format(self.start_iteration))

        self.s_model.train()
        for iteration, (images, targets, sample_names) in enumerate(
            self.train_loader, start=self.start_iteration + 1
        ):
            self.current_iteration = iteration

            images = images.to(self.device)
            targets = targets.long().to(self.device)
            
            with torch.no_grad():
                t_outputs = self.t_model(images)

            s_outputs = self.s_model(images)

            raw_teacher_logits = t_outputs[0]
            teacher_kd_logits = raw_teacher_logits
            if self.args.teacher_output_temp != 1.0:
                teacher_kd_logits = teacher_kd_logits / self.args.teacher_output_temp

            covar_weight = None
            temperature_map = None
            reliability_map = None
            covar_valid_mask = None
            rtc_maps = None
            needs_masked_kd = self.args.kd_loss_mode == 'masked' or self.args.use_covar
            if needs_masked_kd and self.args.lambda_kd != 0.:
                with torch.no_grad():
                    valid_mask = (targets != self.args.ignore_label)
                    if self.args.use_covar and self.args.covar_temp_mode == 'rtc':
                        rtc_maps = build_rtc_temperature_map(
                            raw_teacher_logits,
                            teacher_kd_logits,
                            valid_mask,
                            self.rtc_cdf,
                            self.rtc_config,
                            shuffle=self.args.rtc_shuffle,
                            reverse_routing=self.args.rtc_reverse_routing,
                        )
                        temperature_map = rtc_maps.temperature
                        reliability_map = rtc_maps.reliability
                        covar_valid_mask = rtc_maps.valid_mask
                    elif self.args.use_covar and self.args.covar_temp_mode == 'newton':
                        temperature_map, reliability_map, covar_valid_mask, _, _ = \
                            newton_covar_temperature_map(
                                teacher_kd_logits,
                                valid_mask,
                                self.covar_newton_config,
                            )
                    elif self.args.use_covar:
                        covar_weight = self.get_covar_weight(
                            F.softmax(teacher_kd_logits, dim=1),
                            valid_mask,
                        )
                    else:
                        covar_valid_mask = F.interpolate(
                            valid_mask.float().unsqueeze(1),
                            size=s_outputs[0].shape[-2:],
                            mode='nearest',
                        ).squeeze(1) > 0.5
                        temperature_map = torch.full(
                            covar_valid_mask.shape,
                            float(self.args.kd_temperature),
                            device=s_outputs[0].device,
                            dtype=s_outputs[0].dtype,
                        )
            
            if self.args.aux:
                task_loss = self.criterion(s_outputs[0], targets) + 0.4 * self.criterion(s_outputs[1], targets)
            else:
                task_loss = self.criterion(s_outputs[0], targets)
            
            kd_loss = torch.tensor(0.).to(self.device)
            adv_G_loss = torch.tensor(0.).to(self.device)
            adv_D_loss = torch.tensor(0.).to(self.device)
            skd_loss = torch.tensor(0.).to(self.device)
            cwd_fea_loss = torch.tensor(0.).to(self.device)
            cwd_logit_loss = torch.tensor(0.).to(self.device)
            ifv_loss = torch.tensor(0.).to(self.device)
            fitnet_loss = torch.tensor(0.).to(self.device)
            at_loss = torch.tensor(0.).to(self.device)
            psd_loss = torch.tensor(0.).to(self.device)
            csd_loss = torch.tensor(0.).to(self.device)
            

            if self.args.lambda_adv != 0.:
                adv_G_loss = self.args.lambda_adv * self.criterion_adv_for_G(self.D_model(s_outputs[0]))

            if self.args.lambda_d != 0.:
                adv_D_loss = self.args.lambda_d * (self.criterion_adv(
                    self.D_model(s_outputs[0].detach()),
                    self.D_model(t_outputs[0].detach())))
            
            if self.args.lambda_kd != 0.:
                if rtc_maps is not None:
                    kd_loss = self.args.lambda_kd * masked_temperature_kd_loss(
                        s_outputs[0],
                        teacher_kd_logits,
                        temperature_map,
                        covar_valid_mask,
                        temperature_power=self.rtc_config.kd_temperature_power,
                    )
                elif self.args.kd_loss_mode == 'masked' and not self.args.use_covar:
                    kd_loss = self.args.lambda_kd * masked_temperature_kd_loss(
                        s_outputs[0],
                        teacher_kd_logits,
                        temperature_map,
                        covar_valid_mask,
                        temperature_power=self.args.covar_kd_temp_power,
                    )
                elif self.args.use_covar and temperature_map is not None:
                    kd_loss = self.args.lambda_kd * covar_temperature_kd_loss(
                        s_outputs[0],
                        teacher_kd_logits,
                        temperature_map,
                        covar_valid_mask,
                        temperature_power=self.covar_newton_config.kd_temperature_power,
                    )
                elif self.args.use_covar and covar_weight is not None:
                    kd_loss = self.args.lambda_kd * self.covar_weighted_kd_loss(
                        s_outputs[0], teacher_kd_logits, covar_weight)
                else:
                    kd_loss = self.args.lambda_kd * self.criterion_kd(s_outputs[0], teacher_kd_logits)
            if self.args.lambda_skd != 0:
                skd_loss = self.args.lambda_skd * self.criterion_skd(s_outputs[-1], t_outputs[-1])
            if self.args.lambda_cwd_fea != 0:
                cwd_fea_loss = self.args.lambda_cwd_fea * self.criterion_cwd(s_outputs[-1], t_outputs[-1])
            if self.args.lambda_cwd_logit != 0:
                cwd_logit_loss = self.args.lambda_cwd_logit * self.criterion_cwd(s_outputs[0], t_outputs[0])
            if self.args.lambda_ifv != 0:
                ifv_loss = self.args.lambda_ifv * self.criterion_ifv(s_outputs[-1], t_outputs[-1], targets)
            if self.args.lambda_fitnet != 0:
                fitnet_loss = self.args.lambda_fitnet * self.criterion_fitnet(s_outputs[-1], t_outputs[-1])
            if self.args.lambda_at != 0:
                at_loss = self.args.lambda_at * self.criterion_at(s_outputs[-1], t_outputs[-1])
            if self.args.lambda_psd != 0. and self.args.lambda_csd != 0.:  
                feat_s_list = [s_outputs[-2], s_outputs[-1], s_outputs[0]]
                feat_t_list = [t_outputs[-2], t_outputs[-1], t_outputs[0]]
                psd_loss, csd_loss = self.criterion_dsd(feat_s_list, feat_t_list)
                psd_loss = self.args.lambda_psd * psd_loss
                csd_loss = self.args.lambda_csd * csd_loss

            losses = task_loss + kd_loss + adv_G_loss + \
                        skd_loss + cwd_fea_loss + cwd_logit_loss +\
                        ifv_loss + at_loss + fitnet_loss +\
                        psd_loss + csd_loss 
            D_losses = adv_D_loss
            if rtc_maps is not None or self.args.kd_loss_mode == 'masked':
                local_nonfinite = not bool(torch.isfinite(losses).item())
                if self.use_adv:
                    local_nonfinite = (
                        local_nonfinite
                        or not bool(torch.isfinite(D_losses).item())
                    )
                failure_flags = torch.zeros(3, device=self.device)
                failure_flags[0] = float(local_nonfinite)
                if rtc_maps is not None and not rtc_maps.shuffled:
                    reliable_active = (
                        rtc_maps.valid_mask & (rtc_maps.gate_reliable > 0)
                    )
                    unreliable_active = (
                        rtc_maps.valid_mask & (rtc_maps.gate_unreliable > 0)
                    )
                    failure_flags[1] = float(
                        bool(reliable_active.any().item())
                        and bool(
                            (rtc_maps.temperature[reliable_active]
                             > self.rtc_config.neutral_temperature + 1e-5)
                            .any()
                            .item()
                        )
                    )
                    failure_flags[2] = float(
                        bool(unreliable_active.any().item())
                        and bool(
                            (rtc_maps.temperature[unreliable_active]
                             < self.rtc_config.neutral_temperature - 1e-5)
                            .any()
                            .item()
                        )
                    )
                if get_world_size() > 1:
                    dist.all_reduce(failure_flags, op=dist.ReduceOp.MAX)
                if bool((failure_flags[0] > 0).item()):
                    raise FloatingPointError(
                        'Non-finite generator or discriminator loss in '
                        'RTC/masked KD path on at least one rank'
                    )
                if bool((failure_flags[1] > 0).item()):
                    raise AssertionError(
                        'RTC reliable branch produced T > T0 on at least one rank'
                    )
                if bool((failure_flags[2] > 0).item()):
                    raise AssertionError(
                        'RTC unreliable branch produced T < T0 on at least one rank'
                    )

            lr = self.adjust_lr(
                base_lr=self.args.lr,
                iter=iteration - 1,
                max_iter=self.args.max_iterations,
                power=0.9,
            )
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            if self.use_adv:
                self.D_optimizer.zero_grad()
                D_losses.backward()
                self.D_optimizer.step()

            task_loss_reduced = self.reduce_mean_tensor(task_loss)
            kd_loss_reduced = self.reduce_mean_tensor(kd_loss)
            adv_G_loss_reduced = self.reduce_mean_tensor(adv_G_loss)
            skd_loss_reduced = self.reduce_mean_tensor(skd_loss)
            cwd_fea_loss_reduced = self.reduce_mean_tensor(cwd_fea_loss)
            cwd_logit_loss_reduced = self.reduce_mean_tensor(cwd_logit_loss)
            ifv_loss_reduced = self.reduce_mean_tensor(ifv_loss)
            at_loss_reduced = self.reduce_mean_tensor(at_loss)
            fitnet_loss_reduced = self.reduce_mean_tensor(fitnet_loss)
            psd_loss_reduced = self.reduce_mean_tensor(psd_loss)
            csd_loss_reduced = self.reduce_mean_tensor(csd_loss)
            
            
            D_losses_reduced = self.reduce_mean_tensor(D_losses)
            elapsed_iterations = max(iteration - self.start_iteration, 1)
            eta_seconds = (
                (time.time() - start_time) / elapsed_iterations
            ) * (self.args.max_iterations - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            rtc_diagnostics = None
            if iteration % log_per_iters == 0 and rtc_maps is not None:
                rtc_diagnostics = self.aggregate_rtc_diagnostics(rtc_maps)

            if iteration % log_per_iters == 0 and save_to_disk:
                log_message = (
                    "Iters: {:d}/{:d} || Lr: {:.6f} || Task Loss: {:.4f} || KD Loss: {:.4f}" \
                    "|| Adv_G Loss: {:.4f} || Adv_D Loss: {:.4f}" \
                    "|| skd_loss: {:.4f} || cwd_fea_loss: {:.4f} || cwd_logit_loss: {:.4f} " \
                        "|| ifv_loss: {:.4f} || at_loss: {:.4f} || fitnet_loss: {:.4f} " \
                        "|| psd_loss: {:.4f} || csd_loss: {:.4f} " \
                        "|| Cost Time: {} || Estimated Time: {}".format(
                        iteration, self.args.max_iterations, self.optimizer.param_groups[0]['lr'],
                        task_loss_reduced.item(),
                        kd_loss_reduced.item(), 
                        adv_G_loss_reduced.item(),
                        D_losses_reduced.item(), 
                        skd_loss_reduced.item(),
                        cwd_fea_loss_reduced.item(),
                        cwd_logit_loss_reduced.item(),
                        ifv_loss_reduced.item(),
                        at_loss_reduced.item(),
                        fitnet_loss_reduced.item(),
                        psd_loss_reduced.item(),
                        csd_loss_reduced.item(),
                        str(datetime.timedelta(seconds=int(time.time() - start_time))), 
                        eta_string))
                if rtc_diagnostics is not None:
                    log_message += (
                        " || RTC T mean/hmean/q10/q50/q90/p95: "
                        "{temperature_mean:.4f}/{temperature_harmonic_mean:.4f}/"
                        "{temperature_q10:.4f}/{temperature_q50:.4f}/"
                        "{temperature_q90:.4f}/{temperature_p95:.4f}"
                        " || RTC endpoints R/T0/U: "
                        "{temperature_reliable_endpoint_rate:.4f}/"
                        "{temperature_neutral_rate:.4f}/"
                        "{temperature_unreliable_endpoint_rate:.4f}"
                        " || RTC R/U cov: {reliable_coverage:.4f}/{unreliable_coverage:.4f}"
                        " || RTC gate R/U: {reliable_gate_mean:.4f}/{unreliable_gate_mean:.4f}"
                        " || RTC residual applicable/shuffled: "
                        "{residual_is_applicable:.0f}/{shuffled:.0f}"
                        " || RTC finite/fallback/tie: "
                        "{finite_rate:.6f}/{fallback_rate:.6f}/{tie_rate:.6f}"
                        " || RTC effective teacher T mean: "
                        "{effective_teacher_temperature_mean:.4f}"
                        " || RTC valid pixels: {valid_count:.0f}"
                        " || CDF sha256: {cdf}".format(
                            cdf=self.rtc_cdf.checksum_sha256[:12],
                            **rtc_diagnostics,
                        )
                    )
                    if rtc_diagnostics['residual_is_applicable'] >= 0.5:
                        log_message += (
                            " || RTC residual active/mean/p95/max: "
                            "{target_residual_active_count:.0f}/"
                            "{target_residual_mean:.6f}/{target_residual_p95:.6f}/"
                            "{target_residual_max:.6f}".format(**rtc_diagnostics)
                        )
                    else:
                        log_message += " || RTC residual: N/A (shuffled map)"
                elif temperature_map is not None and reliability_map is not None:
                    valid_temperatures = temperature_map[covar_valid_mask]
                    valid_reliability = reliability_map[covar_valid_mask]
                    if valid_temperatures.numel() > 0:
                        log_message += (
                            " || CoVar T mean/min/max: {:.4f}/{:.4f}/{:.4f} || r_mean: {:.4f}".format(
                                valid_temperatures.mean().item(),
                                valid_temperatures.min().item(),
                                valid_temperatures.max().item(),
                                valid_reliability.mean().item(),
                            )
                        )
                elif temperature_map is not None:
                    valid_temperatures = temperature_map[covar_valid_mask]
                    if valid_temperatures.numel() > 0:
                        log_message += (
                            " || Masked KD T mean/min/max: {:.4f}/{:.4f}/{:.4f} || gamma: {:.1f}".format(
                                valid_temperatures.mean().item(),
                                valid_temperatures.min().item(),
                                valid_temperatures.max().item(),
                                self.args.covar_kd_temp_power,
                            )
                        )
                elif self.args.teacher_output_temp != 1.0:
                    log_message += " || Teacher output T: {:.4f}".format(self.args.teacher_output_temp)
                logger.info(log_message)

            if iteration % save_per_iters == 0:
                self.save_checkpoint(is_best=False, iteration=iteration)

            if not self.args.skip_val and iteration % val_per_iters == 0:
                self.validation(step=iteration)
                self.s_model.train()

        self.save_checkpoint(is_best=False, iteration=self.current_iteration)
        total_training_time = time.time() - start_time
        total_training_str = str(datetime.timedelta(seconds=total_training_time))
        completed_iterations = max(self.current_iteration - self.start_iteration, 1)
        logger.info(
            "Total training time: {} ({:.4f}s / it)".format(
                total_training_str, total_training_time / completed_iterations))


    def validation(self, step=None):
        is_best = False
        self.metric.reset()
        if self.args.distributed:
            model = self.s_model.module
        else:
            model = self.s_model
        empty_accelerator_cache(self.device)  # TODO check if it helps
        model.eval()
        logger.info("Start validation, Total sample: {:d}".format(len(self.val_loader)))
        for i, (image, target, filename) in enumerate(self.val_loader):
            image = image.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                outputs = model(image)

            B, H, W = target.size()
            outputs[0] = F.interpolate(outputs[0], (H, W), mode='bilinear', align_corners=True)

            self.metric.update(outputs[0], target)
            pixAcc, mIoU = self.metric.get()
            logger.info(format_sample_validation_log(i + 1, pixAcc, mIoU))
        
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

            logger.info(format_overall_validation_log(
                pixAcc.item() * 100, mIoU * 100
            ))

        new_pred = float(mIoU)
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
        if is_best:
            self.save_checkpoint(is_best=True, iteration=step, save_latest=False)
        synchronize()


def save_npy(array, name):
    """Save Checkpoint"""
    if (args.distributed is not True) or (args.distributed and args.local_rank == 0):
        directory = os.path.expanduser(args.save_dir)
        np.save(os.path.join(directory, name), array)


def save_checkpoint(
    model,
    args,
    is_best=False,
    training_state=None,
    save_latest=True,
):
    """Save legacy model weights plus an optional complete training state."""
    directory = os.path.expanduser(args.save_dir)
    os.makedirs(directory, exist_ok=True)
    filename = 'kd_{}_{}_{}.pth'.format(
        args.student_model, args.student_backbone, args.dataset
    )
    filename = os.path.join(directory, filename)
    model_state = unwrap_module(model).state_dict()

    if save_latest:
        torch.save(model_state, filename)
        if training_state is not None:
            torch.save(training_state, os.path.join(directory, 'training_state_latest.pth'))

    if is_best:
        best_filename = 'kd_{}_{}_{}_best_model.pth'.format(
            args.student_model, args.student_backbone, args.dataset
        )
        best_filename = os.path.join(directory, best_filename)
        torch.save(model_state, best_filename)
        if training_state is not None:
            torch.save(training_state, os.path.join(directory, 'training_state_best.pth'))


if __name__ == '__main__':
    args = parse_args()

    # reference maskrcnn-benchmark
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.num_gpus = num_gpus
    args.distributed = num_gpus > 1
    if args.no_cuda:
        args.device_type = 'cpu'
    args.device = resolve_device_type(args.device_type)
    seed_everything(args.seed, rank=int(os.environ.get("RANK", 0)))
    if args.device == "cuda":
        cudnn.benchmark = False
        set_accelerator_device(args.device, args.local_rank)
    elif args.device == "npu":
        set_accelerator_device(args.device, args.local_rank)
    else:
        args.distributed = False
        args.device = "cpu"
    if args.distributed:
        backend = "hccl" if args.device == "npu" else "nccl"
        torch.distributed.init_process_group(backend=backend, init_method="env://")
        synchronize()

    logger = setup_logger("semantic_segmentation", args.log_dir, get_rank(), filename='{}_{}_{}_log.txt'.format(
        args.student_model, args.teacher_backbone, args.student_backbone, args.dataset))
    logger.info("Using {} process(es) on device {}".format(num_gpus, args.device))
    logger.info(args)

    trainer = Trainer(args)
    trainer.train()
    empty_accelerator_cache(torch.device(args.device))
