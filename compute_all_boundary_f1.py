import os
import time
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt
from concurrent.futures import ProcessPoolExecutor


def boundary_metrics_multi_tol(pred_path, gt_path, tolerances=[1, 2, 3]):
    pred = np.array(Image.open(pred_path).convert('L')) > 0
    gt = np.array(Image.open(gt_path).convert('L')) > 0

    pred_boundary = pred ^ binary_erosion(pred)
    gt_boundary = gt ^ binary_erosion(gt)

    n_pred = pred_boundary.sum()
    n_gt = gt_boundary.sum()

    res = {}
    if n_pred == 0 and n_gt == 0:
        for tol in tolerances:
            res[tol] = (1.0, 1.0, 1.0)
        return res

    if n_pred == 0 or n_gt == 0:
        for tol in tolerances:
            res[tol] = (0.0, 0.0, 0.0)
        return res

    dist_to_gt = distance_transform_edt(~gt_boundary)
    dist_to_pred = distance_transform_edt(~pred_boundary)

    for tol in tolerances:
        pred_match = pred_boundary & (dist_to_gt <= tol)
        gt_match = gt_boundary & (dist_to_pred <= tol)

        p = pred_match.sum() / n_pred
        r = gt_match.sum() / n_gt
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        res[tol] = (float(p), float(r), float(f1))

    return res


def worker(item):
    p_path, g_path, tols = item
    return boundary_metrics_multi_tol(p_path, g_path, tols)


def main():
    outputs_root = '/root/autodl-tmp/DualStrip-Net/outputs'
    tolerances = [1, 2, 3]

    datasets = ['chn6', 'deepglobe', 'mass']
    splits = ['5%', '10%', '20%']

    all_summary = {}

    print("=" * 90)
    print(f"{'Dataset':<12} {'Split':<8} {'Samples':<8} {'Tol=1 BF1':<15} {'Tol=2 BF1':<15} {'Tol=3 BF1':<15}")
    print("=" * 90)

    for ds in datasets:
        all_summary[ds] = {}
        for sp in splits:
            pred_dir = os.path.join(outputs_root, ds, sp)
            if not os.path.isdir(pred_dir):
                continue

            # Locate GT directory
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
                    # Check for _mask.png suffix
                    alt_f = f.replace('.png', '_mask.png')
                    gt_path = os.path.join(gt_dir, alt_f)
                if not os.path.exists(gt_path):
                    print(f"Warning: GT not found for {ds}/{sp}/{f}")
                    continue
                pairs.append((pred_path, gt_path, tolerances))

            if not pairs:
                continue

            with ProcessPoolExecutor(max_workers=32) as executor:
                results = list(executor.map(worker, pairs))

            sp_results = {}
            for tol in tolerances:
                p_list = [r[tol][0] for r in results]
                r_list = [r[tol][1] for r in results]
                f1_list = [r[tol][2] for r in results]
                sp_results[tol] = {
                    'precision': float(np.mean(p_list)),
                    'recall': float(np.mean(r_list)),
                    'f1': float(np.mean(f1_list)),
                }

            all_summary[ds][sp] = {
                'samples': len(pairs),
                'metrics': sp_results,
            }

            f1_t1 = sp_results[1]['f1']
            f1_t2 = sp_results[2]['f1']
            f1_t3 = sp_results[3]['f1']

            print(f"{ds:<12} {sp:<8} {len(pairs):<8} "
                  f"{f1_t1*100:6.2f}% ({f1_t1:.4f})  "
                  f"{f1_t2*100:6.2f}% ({f1_t2:.4f})  "
                  f"{f1_t3*100:6.2f}% ({f1_t3:.4f})")

    print("=" * 90)

    # Detailed breakdown
    print("\n\n" + "=" * 90)
    print("DETAILED BREAKDOWN (Precision / Recall / Boundary F1)")
    print("=" * 90)
    for ds in datasets:
        print(f"\n--- Dataset: {ds.upper()} ---")
        for sp in splits:
            if sp not in all_summary.get(ds, {}):
                continue
            data = all_summary[ds][sp]
            m = data['metrics']
            print(f"  Split: {sp} (Total {data['samples']} samples)")
            for tol in tolerances:
                p = m[tol]['precision'] * 100.0
                r = m[tol]['recall'] * 100.0
                f1 = m[tol]['f1'] * 100.0
                print(f"    - Tolerance = {tol}: Boundary Precision = {p:6.2f}%, Boundary Recall = {r:6.2f}%, Boundary F1 = {f1:6.2f}% ({m[tol]['f1']:.4f})")


if __name__ == '__main__':
    main()
