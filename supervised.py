import argparse
import logging
import os
import pprint
import imageio

import torch
import numpy as np
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from segmentation_models_pytorch import Unet
from model.semseg.patchnet import PatchNet

# from dataset.semi_chn6 import SemiDataset
# from dataset.semi_mass import SemiDataset
from dataset.semi_dg import SemiDataset
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log
from util.dist_helper import setup_distributed

parser = argparse.ArgumentParser(description='Revisiting DualStrip in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, default=None)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def _distributed_available():
    return dist.is_available() and dist.is_initialized()


def _all_reduce_tensor(tensor):
    if _distributed_available():
        dist.all_reduce(tensor)
    return tensor


def evaluate(model, loader, mode, cfg, path=None):
    """Evaluate binary road segmentation.

    Returns:
        mIOU: mean IoU over classes.
        iou_class: per-class IoU.
        F1: road-class F1-score.
        precision: road-class precision.
        recall: road-class recall.
        kappa: Cohen's Kappa coefficient over all valid pixels.
    """
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    device = next(model.parameters()).device
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    road_tp = torch.tensor(0.0, device=device)
    road_fp = torch.tensor(0.0, device=device)
    road_fn = torch.tensor(0.0, device=device)
    confusion_matrix = torch.zeros(cfg['nclass'], cfg['nclass'], dtype=torch.float64, device=device)

    output_dir = None
    if path is not None:
        # Save inference masks to the project-level Outputs directory as requested.
        # Directory format: Outputs/<dataset>/<split-or-save-dir-name>/
        project_root = os.path.dirname(os.path.abspath(__file__))
        split_name = os.path.basename(os.path.normpath(path))
        output_dir = os.path.join(project_root, 'Outputs', cfg['dataset'], split_name)
        os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for img, mask, id in loader:

            img = img.to(device, non_blocking=True)

            if mode == 'sliding_window':
                grid = cfg['crop_size']
                b, _, h, w = img.shape
                final = torch.zeros(b, cfg['nclass'], h, w, device=img.device)
                counter = torch.zeros(b, 1, h, w, device=img.device)
                row = 0
                stride = max(1, int(grid * 2 / 3))
                while row < h:
                    col = 0
                    row_end = min(h, row + grid)
                    row_start = max(0, row_end - grid)
                    while col < w:
                        col_end = min(w, col + grid)
                        col_start = max(0, col_end - grid)
                        pred_window = model(img[:, :, row_start:row_end, col_start:col_end])
                        final[:, :, row_start:row_end, col_start:col_end] += pred_window.softmax(dim=1)
                        counter[:, :, row_start:row_end, col_start:col_end] += 1
                        col += stride
                    row += stride

                pred = (final / counter.clamp_min(1.0)).argmax(dim=1)

            else:
                if mode == 'center_crop':
                    h, w = img.shape[-2:]
                    start_h, start_w = (h - cfg['crop_size']) // 2, (w - cfg['crop_size']) // 2
                    img = img[:, :, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]
                    mask = mask[:, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]

                pred = model(img).argmax(dim=1)
                if cfg['dataset'] == 'mass':
                    pred = pred[:, 2:-2, 2:-2]
                    mask = mask[:, 2:-2, 2:-2]

            if output_dir is not None:
                filename = os.path.join(output_dir, '{}.png'.format(id[0]))
                pre = (pred.squeeze().cpu().numpy() * 255.).astype(np.uint8)
                imageio.imsave(filename, pre)

            pred_np = pred.cpu().numpy()
            mask_np = mask.numpy()

            intersection, union, target = intersectionAndUnion(pred_np, mask_np, cfg['nclass'], 255)

            reduced_intersection = torch.from_numpy(intersection).to(device)
            reduced_union = torch.from_numpy(union).to(device)

            _all_reduce_tensor(reduced_intersection)
            _all_reduce_tensor(reduced_union)

            intersection_meter.update(reduced_intersection.cpu().numpy())
            union_meter.update(reduced_union.cpu().numpy())

            valid = mask_np != 255
            pred_road = (pred_np == 1) & valid
            target_road = (mask_np == 1) & valid

            batch_tp = torch.tensor(np.logical_and(pred_road, target_road).sum(), dtype=torch.float32, device=device)
            batch_fp = torch.tensor(np.logical_and(pred_road, np.logical_not(target_road)).sum(), dtype=torch.float32, device=device)
            batch_fn = torch.tensor(np.logical_and(np.logical_not(pred_road), target_road).sum(), dtype=torch.float32, device=device)

            road_tp += _all_reduce_tensor(batch_tp)
            road_fp += _all_reduce_tensor(batch_fp)
            road_fn += _all_reduce_tensor(batch_fn)

            valid_flat = valid.reshape(-1)
            pred_flat = pred_np.reshape(-1)[valid_flat]
            target_flat = mask_np.reshape(-1)[valid_flat]
            cm = np.bincount(
                cfg['nclass'] * target_flat.astype(np.int64) + pred_flat.astype(np.int64),
                minlength=cfg['nclass'] ** 2,
            ).reshape(cfg['nclass'], cfg['nclass'])
            reduced_cm = torch.from_numpy(cm).to(device=device, dtype=torch.float64)
            confusion_matrix += _all_reduce_tensor(reduced_cm)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    mIOU = np.mean(iou_class)

    precision = road_tp / (road_tp + road_fp + 1e-10) * 100.0
    recall = road_tp / (road_tp + road_fn + 1e-10) * 100.0
    F1 = 2 * precision * recall / (precision + recall + 1e-10)

    total = confusion_matrix.sum()
    observed_acc = torch.trace(confusion_matrix) / (total + 1e-10)
    row_sum = confusion_matrix.sum(dim=1)
    col_sum = confusion_matrix.sum(dim=0)
    expected_acc = (row_sum * col_sum).sum() / ((total * total) + 1e-10)
    kappa = (observed_acc - expected_acc) / (1.0 - expected_acc + 1e-10)

    return mIOU, iou_class, F1.item(), precision.item(), recall.item(), (kappa * 100.0).item()


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))

        writer = SummaryWriter(args.save_path)

        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    model = Unet(encoder_name=cfg['backbone'], classes=cfg.get('nclass', 2))

    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    optimizer = SGD(model.parameters(), lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        output_device=local_rank,
        find_unused_parameters=False,
    )

    if cfg['criterion']['name'] == 'CELoss':
        criterion = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    trainset = SemiDataset(cfg, cfg['dataset'], cfg['data_root'], 'train_l', cfg['crop_size'], args.labeled_id_path)
    valset = SemiDataset(cfg, cfg['dataset'], cfg['data_root'], 'val')

    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=cfg['batch_size'],
                             pin_memory=True, num_workers=2, drop_last=True, sampler=trainsampler)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=2, drop_last=False, sampler=valsampler)

    total_iters = len(trainloader) * cfg['epochs']
    previous_best = 0.0
    epoch = -1

    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'), map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']

        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)

    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        model.train()
        total_loss = AverageMeter()

        trainsampler.set_epoch(epoch)

        for i, (img, mask) in enumerate(trainloader):

            img, mask = img.cuda(local_rank, non_blocking=True), mask.cuda(local_rank, non_blocking=True)
            pred = model(img)
            loss = criterion(pred, mask)

            torch.distributed.barrier()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())

            iters = epoch * len(trainloader) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr

            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss.item(), iters)

            if (i % (max(2, len(trainloader) // 8)) == 0) and (rank == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}'.format(i, total_loss.avg))

        eval_mode = 'sliding_window' if cfg['dataset'] == 'mass' else 'original'
        mIoU, iou_class, F1, precision, recall, kappa = evaluate(model, valloader, eval_mode, cfg)

        road_iou = float(iou_class[1]) if len(iou_class) > 1 else float(mIoU)

        if rank == 0:
            for (cls_idx, iou) in enumerate(iou_class):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                            'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
            logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f} RoadIoU: {:.2f} '
                        'Precision: {:.2f} Recall: {:.2f} F1: {:.2f} Kappa: {:.2f}\n'.format(
                            eval_mode, mIoU, road_iou, precision, recall, F1, kappa))

            writer.add_scalar('eval/mIoU', mIoU, epoch)
            writer.add_scalar('eval/road_IoU', road_iou, epoch)
            writer.add_scalar('eval/precision', precision, epoch)
            writer.add_scalar('eval/recall', recall, epoch)
            writer.add_scalar('eval/F1', F1, epoch)
            writer.add_scalar('eval/kappa', kappa, epoch)
            for i, iou in enumerate(iou_class):
                writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)

        is_best = road_iou > previous_best
        previous_best = max(road_iou, previous_best)
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
                'best_metric': 'road_iou',
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))


if __name__ == '__main__':
    main()
