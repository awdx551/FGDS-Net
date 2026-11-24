import argparse
import copy
import json
import logging
import os
import pprint
import time
import imageio
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt
from skimage.morphology import skeletonize
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
import yaml

from dataset.semi_dg import SemiDataset
from model.semseg.patchnet import PatchNet
from util.classes import CLASSES
from util.utils import count_params, init_log, AverageMeter, intersectionAndUnion
from util.dist_helper import setup_distributed

parser = argparse.ArgumentParser(description='Test DualStrip-Net for Semi-Supervised Road Segmentation')
parser.add_argument('--config', type=str, required=True, help='Path to dataset YAML config')
parser.add_argument('--labeled-id-path', type=str, default=None, help='Path to labeled ids (optional)')
parser.add_argument('--unlabeled-id-path', type=str, default=None, help='Path to unlabeled ids (optional)')
parser.add_argument('--save-path', type=str, required=True, help='Path where checkpoint is saved (e.g. exp/dataset/...)')
parser.add_argument('--checkpoint', type=str, default=None, help='Path to specific checkpoint file (default: save-path/best.pth)')
parser.add_argument('--output-dir', type=str, default=None, help='Path to save output prediction masks')
parser.add_argument('--eval-mode', type=str, default=None, choices=['original', 'sliding_window', 'center_crop'],
                    help='Evaluation mode (default: sliding_window for mass, original for others)')
parser.add_argument('--tolerance', type=int, default=2, help='Pixel tolerance for boundary F1/Precision/Recall calculation')
parser.add_argument('--local_rank', '--local-rank', dest='local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def _distributed_available():
    return dist.is_available() and dist.is_initialized()


def _all_reduce_tensor(tensor):
    if _distributed_available():
        dist.all_reduce(tensor)
    return tensor


def load_checkpoint_compatible(path, map_location='cpu'):
    """Load checkpoints across PyTorch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def boundary_f1(pred_mask, gt_mask, tolerance=2):
    """
    Calculate Boundary Precision, Boundary Recall, and Boundary F1 score.
    Reference implementation matching user specification.
    """
    pred = pred_mask > 0
    # If ground truth contains ignore index (e.g. 255), mask it out
    if np.any(gt_mask == 255) and np.any(gt_mask == 1):
        valid = gt_mask != 255
        gt = (gt_mask == 1) & valid
        pred = pred & valid
    else:
        gt = gt_mask > 0

    pred_boundary = pred ^ binary_erosion(pred)
    gt_boundary = gt ^ binary_erosion(gt)

    n_pred = pred_boundary.sum()
    n_gt = gt_boundary.sum()

    if n_pred == 0 and n_gt == 0:
        return 1.0, 1.0, 1.0

    if n_pred == 0 or n_gt == 0:
        return 0.0, 0.0, 0.0

    dist_to_gt = distance_transform_edt(~gt_boundary)
    dist_to_pred = distance_transform_edt(~pred_boundary)

    pred_match = pred_boundary & (dist_to_gt <= tolerance)
    gt_match = gt_boundary & (dist_to_pred <= tolerance)

    precision = float(pred_match.sum() / n_pred)
    recall = float(gt_match.sum() / n_gt)

    if precision + recall == 0:
        bf1 = 0.0
    else:
        bf1 = float(2 * precision * recall / (precision + recall))

    return precision, recall, bf1


def cldice(pred_mask, gt_mask):
    """
    Calculate skeleton-based clDice metric.
    tprec = sum(pred_skeleton & gt) / sum(pred_skeleton)
    tsens = sum(gt_skeleton & pred) / sum(gt_skeleton)
    clDice = 2 * tprec * tsens / (tprec + tsens)
    """
    pred = pred_mask > 0
    gt = gt_mask > 0

    pred_skeleton = skeletonize(pred)
    gt_skeleton = skeletonize(gt)

    pred_num = pred_skeleton.sum()
    gt_num = gt_skeleton.sum()

    if pred_num == 0 and gt_num == 0:
        return 1.0, 1.0, 1.0

    if pred_num == 0 or gt_num == 0:
        return 0.0, 0.0, 0.0

    tprec = np.sum(pred_skeleton & gt) / pred_num
    tsens = np.sum(gt_skeleton & pred) / gt_num

    if tprec + tsens == 0:
        score = 0.0
    else:
        score = 2 * tprec * tsens / (tprec + tsens)

    return float(tprec), float(tsens), float(score)


def measure_model_complexity(model, input_size=(1, 3, 512, 512), device='cuda'):
    """Measure total parameters, trainable parameters, and FLOPs."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    raw_model = model.module if hasattr(model, 'module') else model
    eval_model = copy.deepcopy(raw_model).to(device).eval()

    flops_val = None
    flops_str = "N/A"
    try:
        from thop import profile, clever_format
        dummy_input = torch.randn(*input_size, device=device)
        flops, _ = profile(eval_model, inputs=(dummy_input,), verbose=False)
        flops_val = float(flops)
        flops_str, _ = clever_format([flops, total_params], "%.3f")
    except Exception as e:
        try:
            from ptflops import get_model_complexity_info
            macs_str, _ = get_model_complexity_info(
                eval_model, input_size[1:], as_strings=True, print_per_layer_stat=False, verbose=False
            )
            flops_str = macs_str
        except Exception:
            flops_str = f"Profiling error: {e}"

    del eval_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'params_m': total_params / 1e6,
        'trainable_params_m': trainable_params / 1e6,
        'flops': flops_val,
        'flops_g': (flops_val / 1e9) if flops_val is not None else None,
        'flops_str': flops_str,
        'input_size': list(input_size),
    }


def evaluate_test(model, loader, mode, cfg, output_dir=None, tolerance=2, logger=None):
    """
    Run full test evaluation, save prediction masks, measure inference time,
    compute 5 training quantitative metrics + 3 boundary metrics + clDice.
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

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    # Warmup GPU
    warmup_count = min(3, len(loader))
    with torch.no_grad():
        for i, (img, _, _) in enumerate(loader):
            if i >= warmup_count:
                break
            img = img.to(device, non_blocking=True)
            if mode == 'sliding_window':
                grid = cfg['crop_size']
                _ = model(img[:, :, :grid, :grid])
            else:
                _ = model(img)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    infer_times = []
    sample_b_precisions = []
    sample_b_recalls = []
    sample_b_f1s = []
    sample_tprecs = []
    sample_tsens = []
    sample_cldices = []

    with torch.no_grad():
        for img, mask, img_id in loader:
            img = img.to(device, non_blocking=True)
            img_id_str = str(img_id[0]) if isinstance(img_id, (list, tuple)) else str(img_id)
            base_id = os.path.splitext(img_id_str)[0]

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_start = time.perf_counter()

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

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_end = time.perf_counter()
            infer_times.append(t_end - t_start)

            # Save predicted mask as uint8 PNG (0 for background, 255 for road)
            if output_dir is not None:
                mask_filename = os.path.join(output_dir, f"{base_id}.png")
                pred_binary = (pred.squeeze().cpu().numpy() * 255.0).astype(np.uint8)
                imageio.imsave(mask_filename, pred_binary)

            pred_np = pred.cpu().numpy()
            mask_np = mask.numpy()

            # Global confusion & IoU calculation
            intersection, union, _ = intersectionAndUnion(pred_np, mask_np, cfg['nclass'], 255)
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

            # Sample-wise boundary metrics
            b_p, b_r, b_f1 = boundary_f1(pred_np.squeeze(), mask_np.squeeze(), tolerance=tolerance)
            sample_b_precisions.append(b_p)
            sample_b_recalls.append(b_r)
            sample_b_f1s.append(b_f1)

            # Sample-wise clDice metric
            tp_c, ts_c, cld_s = cldice(pred_np.squeeze(), mask_np.squeeze())
            sample_tprecs.append(tp_c)
            sample_tsens.append(ts_c)
            sample_cldices.append(cld_s)

    # Compute aggregated metrics
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    mIoU = float(np.mean(iou_class))
    road_iou = float(iou_class[1]) if len(iou_class) > 1 else mIoU
    bg_iou = float(iou_class[0]) if len(iou_class) > 0 else 0.0

    precision = float((road_tp / (road_tp + road_fp + 1e-10) * 100.0).item())
    recall = float((road_tp / (road_tp + road_fn + 1e-10) * 100.0).item())
    F1 = float(2 * precision * recall / (precision + recall + 1e-10))

    total = confusion_matrix.sum()
    observed_acc = torch.trace(confusion_matrix) / (total + 1e-10)
    row_sum = confusion_matrix.sum(dim=1)
    col_sum = confusion_matrix.sum(dim=0)
    expected_acc = (row_sum * col_sum).sum() / ((total * total) + 1e-10)
    kappa = float(((observed_acc - expected_acc) / (1.0 - expected_acc + 1e-10) * 100.0).item())
    overall_acc = float((observed_acc * 100.0).item())

    # Aggregate boundary metrics across all processes if distributed
    if _distributed_available():
        tensor_bp = torch.tensor(sample_b_precisions, device=device)
        tensor_br = torch.tensor(sample_b_recalls, device=device)
        tensor_bf1 = torch.tensor(sample_b_f1s, device=device)
        tensor_cld = torch.tensor(sample_cldices, device=device)
        tensor_tprec = torch.tensor(sample_tprecs, device=device)
        tensor_tsens = torch.tensor(sample_tsens, device=device)

        world_size = dist.get_world_size()
        gathered_bp = [torch.zeros_like(tensor_bp) for _ in range(world_size)]
        gathered_br = [torch.zeros_like(tensor_br) for _ in range(world_size)]
        gathered_bf1 = [torch.zeros_like(tensor_bf1) for _ in range(world_size)]
        gathered_cld = [torch.zeros_like(tensor_cld) for _ in range(world_size)]
        gathered_tprec = [torch.zeros_like(tensor_tprec) for _ in range(world_size)]
        gathered_tsens = [torch.zeros_like(tensor_tsens) for _ in range(world_size)]

        dist.all_gather(gathered_bp, tensor_bp)
        dist.all_gather(gathered_br, tensor_br)
        dist.all_gather(gathered_bf1, tensor_bf1)
        dist.all_gather(gathered_cld, tensor_cld)
        dist.all_gather(gathered_tprec, tensor_tprec)
        dist.all_gather(gathered_tsens, tensor_tsens)

        mean_b_precision = float(torch.cat(gathered_bp).mean().item()) * 100.0
        mean_b_recall = float(torch.cat(gathered_br).mean().item()) * 100.0
        mean_b_f1 = float(torch.cat(gathered_bf1).mean().item()) * 100.0
        mean_cldice = float(torch.cat(gathered_cld).mean().item()) * 100.0
        mean_tprec = float(torch.cat(gathered_tprec).mean().item()) * 100.0
        mean_tsens = float(torch.cat(gathered_tsens).mean().item()) * 100.0
    else:
        mean_b_precision = float(np.mean(sample_b_precisions)) * 100.0
        mean_b_recall = float(np.mean(sample_b_recalls)) * 100.0
        mean_b_f1 = float(np.mean(sample_b_f1s)) * 100.0
        mean_cldice = float(np.mean(sample_cldices)) * 100.0
        mean_tprec = float(np.mean(sample_tprecs)) * 100.0
        mean_tsens = float(np.mean(sample_tsens)) * 100.0

    avg_infer_time_ms = float(np.mean(infer_times) * 1000.0) if len(infer_times) > 0 else 0.0
    fps = float(1000.0 / avg_infer_time_ms) if avg_infer_time_ms > 0 else 0.0

    results = {
        'total_samples': len(loader.dataset),
        'eval_mode': mode,
        'avg_infer_time_ms': avg_infer_time_ms,
        'fps': fps,
        # Training Quantitative Metrics
        'mIoU': mIoU,
        'road_iou': road_iou,
        'bg_iou': bg_iou,
        'iou_class': [float(x) for x in iou_class],
        'precision': precision,
        'recall': recall,
        'f1': F1,
        'kappa': kappa,
        'overall_acc': overall_acc,
        # 3 Boundary Metrics
        'boundary_precision': mean_b_precision,
        'boundary_recall': mean_b_recall,
        'boundary_f1': mean_b_f1,
        'tolerance': tolerance,
        # Topology clDice Metric
        'cldice': mean_cldice,
        'tprec': mean_tprec,
        'tsens': mean_tsens,
    }
    return results


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    # Avoid HuggingFace weight downloads during evaluation
    cfg['encoder_weights'] = None

    # Determine rank & world size
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank, world_size = setup_distributed(port=args.port)
        distributed = True
    else:
        rank, world_size = 0, 1
        distributed = False
        if torch.cuda.is_available():
            torch.cuda.set_device(args.local_rank)

    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    # Determine output mask directory: ./outputs/{dataset}/{split}/
    split_name = os.path.basename(os.path.normpath(args.save_path))
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join("./outputs", cfg['dataset'], split_name)

    if rank == 0:
        os.makedirs(args.save_path, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        all_args = {**cfg, **vars(args), 'output_dir': output_dir, 'ngpus': world_size}
        logger.info('=' * 80)
        logger.info('Test Configurations:\n{}'.format(pprint.pformat(all_args)))
        logger.info('=' * 80)

    cudnn.enabled = True
    cudnn.benchmark = True

    # Initialize model
    model = PatchNet(cfg)

    # Complexity estimation (Params and FLOPs) on rank 0
    complexity_info = {}
    if rank == 0:
        complexity_info = measure_model_complexity(
            model,
            input_size=(1, 3, cfg['crop_size'], cfg['crop_size']),
            device=f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
        )
        logger.info(f"Model Complexity Analysis:")
        logger.info(f"  - Total Parameters:      {complexity_info['params_m']:.2f} M ({complexity_info['total_params']:,})")
        logger.info(f"  - Trainable Parameters:  {complexity_info['trainable_params_m']:.2f} M ({complexity_info['trainable_params']:,})")
        logger.info(f"  - FLOPs (@ {cfg['crop_size']}x{cfg['crop_size']}):     {complexity_info['flops_str']}")
        logger.info('-' * 80)

    if torch.cuda.is_available():
        if distributed:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(local_rank)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            output_device=local_rank,
            find_unused_parameters=False,
        )

    # Load dataset
    testset = SemiDataset(cfg, cfg['dataset'], cfg['data_root'], 'test')
    if distributed:
        testsampler = torch.utils.data.distributed.DistributedSampler(testset, shuffle=False)
        testloader = DataLoader(testset, batch_size=1, pin_memory=True, num_workers=1,
                                drop_last=False, sampler=testsampler)
    else:
        testloader = DataLoader(testset, batch_size=1, shuffle=False, pin_memory=True,
                                num_workers=2, drop_last=False)

    # Load Checkpoint
    checkpoint_path = args.checkpoint if args.checkpoint is not None else os.path.join(args.save_path, 'best.pth')
    if not os.path.exists(checkpoint_path):
        # Fallback to latest.pth if best.pth not found
        alt_path = os.path.join(args.save_path, 'latest.pth')
        if os.path.exists(alt_path):
            checkpoint_path = alt_path

    if rank == 0:
        logger.info(f"Loading checkpoint from: {checkpoint_path}")

    checkpoint = load_checkpoint_compatible(checkpoint_path, map_location='cpu')
    raw_state = checkpoint['model'] if 'model' in checkpoint else checkpoint
    if not distributed:
        # Strip 'module.' prefix if model is single-GPU but checkpoint was saved with DDP
        clean_state = {k.replace('module.', ''): v for k, v in raw_state.items()}
        model.load_state_dict(clean_state)
    else:
        # If model is DDP, ensure keys have 'module.' prefix
        ddp_state = {}
        for k, v in raw_state.items():
            key = k if k.startswith('module.') else f'module.{k}'
            ddp_state[key] = v
        model.load_state_dict(ddp_state)

    if rank == 0:
        logger.info('Successfully loaded model weights from checkpoint.\n')

    # Determine eval mode
    if args.eval_mode is not None:
        eval_mode = args.eval_mode
    else:
        eval_mode = 'sliding_window' if cfg['dataset'] == 'mass' else 'original'

    # Run full test evaluation
    results = evaluate_test(
        model=model,
        loader=testloader,
        mode=eval_mode,
        cfg=cfg,
        output_dir=output_dir,
        tolerance=args.tolerance,
        logger=logger
    )

    if rank == 0:
        results.update(complexity_info)
        results['dataset'] = cfg['dataset']
        results['split'] = split_name
        results['checkpoint'] = checkpoint_path
        results['output_dir'] = output_dir

        # Formatted Summary Display
        class_names = CLASSES.get(cfg['dataset'], [f"Class_{i}" for i in range(cfg['nclass'])])
        summary_text = f"""
================================================================================
=========================== TEST EVALUATION SUMMARY ===========================
================================================================================
Dataset:                  {cfg['dataset']}
Split / Ratio:            {split_name}
Checkpoint:               {checkpoint_path}
Mask Output Directory:    {output_dir}
Evaluation Mode:          {eval_mode}
Total Test Samples:       {results['total_samples']}
--------------------------------------------------------------------------------
[1. Model Complexity & Inference Speed]
  - Total Parameters:     {results.get('params_m', 0.0):.2f} M ({results.get('total_params', 0):,})
  - Trainable Parameters: {results.get('trainable_params_m', 0.0):.2f} M ({results.get('trainable_params', 0):,})
  - FLOPs (@ {cfg['crop_size']}x{cfg['crop_size']}):      {results.get('flops_str', 'N/A')}
  - Avg Inference Time:   {results['avg_infer_time_ms']:.2f} ms / image
  - Inference FPS:        {results['fps']:.2f} frames/sec
--------------------------------------------------------------------------------
[2. Training Quantitative Metrics (5 Core Metrics)]
  - Mean IoU (mIoU):      {results['mIoU']:.2f}%
  - Road IoU (Class 1):   {results['road_iou']:.2f}%
  - Background IoU:       {results['bg_iou']:.2f}%
  - Road Precision:       {results['precision']:.2f}%
  - Road Recall:          {results['recall']:.2f}%
  - Road F1-Score:        {results['f1']:.2f}%
  - Cohen's Kappa:        {results['kappa']:.2f}%
  - Overall Accuracy:     {results['overall_acc']:.2f}%
--------------------------------------------------------------------------------
[3. Per-Class IoU Breakdown]"""

        for cls_idx, iou_val in enumerate(results['iou_class']):
            c_name = class_names[cls_idx] if cls_idx < len(class_names) else f"Class_{cls_idx}"
            summary_text += f"\n  - Class [{cls_idx}] {c_name:<12}: {iou_val:.2f}%"

        summary_text += f"""
--------------------------------------------------------------------------------
[4. Boundary Metrics (Tolerance = {args.tolerance} px)]
  - Boundary Precision:   {results['boundary_precision']:.2f}% ({results['boundary_precision']/100.0:.4f})
  - Boundary Recall:      {results['boundary_recall']:.2f}% ({results['boundary_recall']/100.0:.4f})
  - Boundary F1-Score:    {results['boundary_f1']:.2f}% ({results['boundary_f1']/100.0:.4f})
--------------------------------------------------------------------------------
[5. Topological Connectivity Metric (clDice)]
  - clDice Score:         {results['cldice']:.2f}% ({results['cldice']/100.0:.4f})
  - Topological Prec:     {results['tprec']:.2f}% ({results['tprec']/100.0:.4f})
  - Topological Sens:     {results['tsens']:.2f}% ({results['tsens']/100.0:.4f})
================================================================================
"""
        logger.info(summary_text)

        # Save summary to files
        txt_path = os.path.join(output_dir, 'metrics_summary.txt')
        json_path = os.path.join(output_dir, 'metrics_summary.json')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        # Also write a copy to checkpoint directory
        ckpt_summary_txt = os.path.join(args.save_path, 'test_metrics_summary.txt')
        with open(ckpt_summary_txt, 'w', encoding='utf-8') as f:
            f.write(summary_text)

        logger.info(f"Results successfully saved to:\n  -> {txt_path}\n  -> {json_path}")


if __name__ == '__main__':
    main()
