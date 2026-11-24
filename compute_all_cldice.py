import os
import time
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize
from concurrent.futures import ProcessPoolExecutor


def cldice(pred, gt):
    """
    Compute skeleton-based clDice metric.
    Reference implementation:
    tprec = sum(pred_skeleton & gt) / sum(pred_skeleton)
    tsens = sum(gt_skeleton & pred) / sum(gt_skeleton)
    clDice = 2 * tprec * tsens / (tprec + tsens)
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

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


def worker(item):
    pred_path, gt_path = item
    gt = np.array(Image.open(gt_path).convert("L")) > 127
    pred = np.array(Image.open(pred_path).convert("L")) > 127
    return cldice(pred, gt)


def main():
    outputs_root = '/root/autodl-tmp/DualStrip-Net/outputs'
    datasets = ['chn6', 'deepglobe', 'mass']
    splits = ['5%', '10%', '20%']

    all_summary = {}

    print("=" * 95)
    print(f"{'Dataset':<12} {'Split':<8} {'Samples':<8} {'TPrec (Topology Prec)':<22} {'TSens (Topology Sens)':<22} {'clDice Score':<18}")
    print("=" * 95)

    for ds in datasets:
        all_summary[ds] = {}
        for sp in splits:
            pred_dir = os.path.join(outputs_root, ds, sp)
            if not os.path.isdir(pred_dir):
                continue

            gt_dir = os.path.join('/root/autodl-tmp/DualStrip-Net', ds, 'test', 'masks')
            if not os.path.exists(gt_dir):
                gt_dir = os.path.join('/root/autodl-tmp/DualStrip-Net', ds, 'val', 'masks')

            pairs = []
            for f in sorted(os.listdir(pred_dir)):
                if not f.endswith('.png'):
                    continue
                pred_path = os.path.join(pred_dir, f)
                gt_path = os.path.join(gt_dir, f)
                if not os.path.exists(gt_path):
                    alt_f = f.replace('.png', '_mask.png')
                    gt_path = os.path.join(gt_dir, alt_f)
                if not os.path.exists(gt_path):
                    print(f"Warning: GT not found for {ds}/{sp}/{f}")
                    continue
                pairs.append((pred_path, gt_path))

            if not pairs:
                continue

            with ProcessPoolExecutor(max_workers=32) as executor:
                results = list(executor.map(worker, pairs))

            tprec_list = [r[0] for r in results]
            tsens_list = [r[1] for r in results]
            cldice_list = [r[2] for r in results]

            mean_tprec = float(np.mean(tprec_list))
            mean_tsens = float(np.mean(tsens_list))
            mean_cldice = float(np.mean(cldice_list))
            std_cldice = float(np.std(cldice_list))

            all_summary[ds][sp] = {
                'samples': len(pairs),
                'tprec': mean_tprec,
                'tsens': mean_tsens,
                'cldice': mean_cldice,
                'std_cldice': std_cldice,
            }

            print(f"{ds:<12} {sp:<8} {len(pairs):<8} "
                  f"{mean_tprec*100:6.2f}% ({mean_tprec:.4f})       "
                  f"{mean_tsens*100:6.2f}% ({mean_tsens:.4f})       "
                  f"{mean_cldice*100:6.2f}% ({mean_cldice:.4f})")

    print("=" * 95)

    # Detailed dataset breakdown
    print("\n" + "=" * 95)
    print("DETAILED clDice REPORT (Mean & Std & Precision & Sensitivity)")
    print("=" * 95)
    for ds in datasets:
        print(f"\n--- Dataset: {ds.upper()} ---")
        for sp in splits:
            if sp not in all_summary.get(ds, {}):
                continue
            data = all_summary[ds][sp]
            print(f"  Split: {sp} (Total {data['samples']} samples)")
            print(f"    - clDice Score        : {data['cldice']*100:.2f}% (Mean: {data['cldice']:.4f}, Std: {data['std_cldice']:.4f})")
            print(f"    - Topological Prec (TPrec): {data['tprec']*100:.2f}% ({data['tprec']:.4f})")
            print(f"    - Topological Sens (TSens): {data['tsens']*100:.2f}% ({data['tsens']:.4f})")


if __name__ == '__main__':
    main()
