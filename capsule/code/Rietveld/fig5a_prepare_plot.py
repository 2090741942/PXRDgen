import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure


def build_structure(frac_coords, atom_types, lengths, angles):
    lattice = Lattice.from_parameters(
        float(lengths[0]), float(lengths[1]), float(lengths[2]),
        float(angles[0]), float(angles[1]), float(angles[2])
    )
    return Structure(
        lattice=lattice,
        species=list(atom_types),
        coords=np.array(frac_coords),
        coords_are_cartesian=False,
    )


def extract_gt_batch(gt):
    if isinstance(gt, dict):
        return (
            gt["frac_coords"],
            gt["atom_types"],
            gt["lengths"],
            gt["angles"],
            gt["num_atoms"],
        )
    else:
        return (
            gt.frac_coords,
            gt.atom_types,
            gt.lengths,
            gt.angles,
            gt.num_atoms,
        )


def safe_rms(sm, s1, s2):
    try:
        rms = sm.get_rms_dist(s1, s2)
        return None if rms is None else float(rms[0])
    except Exception:
        return None


def prepare(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = torch.load(args.pt_path, map_location="cpu")

    if "input_data_batch" not in data:
        raise ValueError("pt file does not contain input_data_batch, cannot build ground truth structures.")

    gt_frac, gt_types, gt_lengths, gt_angles, gt_num_atoms = extract_gt_batch(data["input_data_batch"])

    frac_coords = data["frac_coords"]
    atom_types = data["atom_types"]
    lengths = data["lengths"]
    angles = data["angles"]
    num_atoms = data["num_atoms"]

    multi_eval = args.multi_eval

    # 兼容：
    # multi_eval=True  -> [num_eval, N, ...]
    # multi_eval=False -> [1, N, ...] 或 [N, ...]
    if multi_eval:
        num_evals = frac_coords.shape[0]
        num_materials = frac_coords.shape[1]
    else:
        if len(frac_coords.shape) == 3:
            num_evals = 1
            num_materials = frac_coords.shape[0]
        else:
            num_evals = 1
            num_materials = frac_coords.shape[0]

    sm = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)

    manifest = []

    for i in range(num_materials):
        gt_structure = build_structure(
            gt_frac[i], gt_types[i], gt_lengths[i], gt_angles[i]
        )

        best_eval_idx = None
        best_rms = None
        best_pred_structure = None

        for j in range(num_evals):
            try:
                if multi_eval:
                    pred_structure = build_structure(
                        frac_coords[j][i],
                        atom_types[j][i],
                        lengths[j][i],
                        angles[j][i],
                    )
                else:
                    pred_structure = build_structure(
                        frac_coords[i],
                        atom_types[i],
                        lengths[i],
                        angles[i],
                    )
            except Exception:
                continue

            rms = safe_rms(sm, gt_structure, pred_structure)
            if rms is None:
                continue

            if best_rms is None or rms < best_rms:
                best_rms = rms
                best_eval_idx = j
                best_pred_structure = pred_structure

        if best_pred_structure is None:
            # 如果一个都没匹配上，就跳过这个样本
            continue

        mat_dir = out_dir / f"mat_{i:05d}"
        mat_dir.mkdir(exist_ok=True)

        pre_path = mat_dir / "pre.cif"
        tar_path = mat_dir / "tar.cif"

        best_pred_structure.to(filename=str(pre_path))
        gt_structure.to(filename=str(tar_path))

        manifest.append({
            "mat_id": i,
            "best_eval_idx": best_eval_idx,
            "oracle_rms_before": best_rms,
            "pre_cif": str(pre_path),
            "tar_cif": str(tar_path),
            "work_dir": str(mat_dir),
        })

    df = pd.DataFrame(manifest)
    manifest_path = out_dir / "fig5a_manifest.csv"
    df.to_csv(manifest_path, index=False)

    print(f"manifest saved to {manifest_path}")
    print(f"num selected materials: {len(df)}")


def plot(args):
    df = pd.read_csv(args.results)

    # 只保留 refinement 成功且 before/after 都有效的样本
    if "refine_success" in df.columns:
        df = df[df["refine_success"] == 1]

    df = df.dropna(subset=["rms_before", "rms_after"])
    df = df[(df["rms_before"] > 0) & (df["rms_after"] > 0)]

    before = np.sort(df["rms_before"].values)
    after = np.sort(df["rms_after"].values)

    n = min(len(before), len(after))
    before = before[:n]
    after = after[:n]
    x = np.arange(1, n + 1)

    plt.figure(figsize=(7, 5.5))
    plt.plot(x, np.log10(after), label="After Rietveld", linewidth=2)
    plt.plot(x, np.log10(before), label="Before Rietveld", linewidth=2)

    plt.xlabel("Counts")
    plt.ylabel("log(RMSE)")
    plt.legend()
    plt.tight_layout()

    save_path = Path(args.out_dir) / "fig5a.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"figure saved to {save_path}")
    print(f"num plotted materials: {n}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p1 = sub.add_parser("prepare")
    p1.add_argument("--pt_path", required=True)
    p1.add_argument("--out_dir", required=True)
    p1.add_argument("--multi_eval", action="store_true")

    p2 = sub.add_parser("plot")
    p2.add_argument("--results", required=True)
    p2.add_argument("--out_dir", required=True)

    args = parser.parse_args()

    if args.cmd == "prepare":
        prepare(args)
    elif args.cmd == "plot":
        plot(args)
    else:
        parser.print_help()