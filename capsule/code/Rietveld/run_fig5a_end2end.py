import os
import sys
import csv
import math
import shutil
import argparse
import traceback
from typing import List, Optional, Tuple, Dict

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher
from collections import Counter

# =========================
# 0. 路径与依赖
# =========================
# 这里按 package 方式导入 GSAS-II，避免 relative import 报错
sys.path.insert(0, '/workspace/g2full/GSAS-II')

from GSASII import GSASIIscriptable as G2sc  # noqa: E402

G2sc.SetPrintLevel("none")

# 让脚本能找到 xrd2struc/scripts/eval_utils.py
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
XRD2STRUC_SCRIPTS_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "xrd2struc", "scripts"))
if XRD2STRUC_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, XRD2STRUC_SCRIPTS_DIR)

from eval_utils import smact_validity, structure_validity, get_crystals_list  # noqa: E402


# =========================
# 1. 与 xrd2struc/compute.py 对齐的数据结构
# =========================
class Crystal:
    def __init__(self, crys_array_dict: Dict):
        self.frac_coords = crys_array_dict["frac_coords"]
        self.atom_types = crys_array_dict["atom_types"]
        self.lengths = crys_array_dict["lengths"]
        self.angles = crys_array_dict["angles"]
        self.dict = crys_array_dict

        self.constructed = False
        self.valid = False
        self.invalid_reason = None
        self.structure = None

        self.get_structure()
        self.get_composition()
        self.get_validity()

    def get_structure(self):
        if min(self.lengths.tolist()) < 0:
            self.constructed = False
            self.invalid_reason = "non_positive_lattice"
            return

        if (
            np.isnan(self.lengths).any()
            or np.isnan(self.angles).any()
            or np.isnan(self.frac_coords).any()
        ):
            self.constructed = False
            self.invalid_reason = "nan_value"
            return

        try:
            self.structure = Structure(
                lattice=Lattice.from_parameters(
                    *(self.lengths.tolist() + self.angles.tolist())
                ),
                species=self.atom_types,
                coords=self.frac_coords,
                coords_are_cartesian=False,
            )
            self.constructed = True
        except Exception:
            self.constructed = False
            self.invalid_reason = "construction_raises_exception"
            return

        if self.structure.volume < 0.1:
            self.constructed = False
            self.invalid_reason = "unrealistically_small_lattice"

    def get_composition(self):
        elem_counter = Counter(self.atom_types)
        composition = [(elem, elem_counter[elem]) for elem in sorted(elem_counter.keys())]
        elems, counts = list(zip(*composition))
        counts = np.array(counts)
        counts = counts / np.gcd.reduce(counts)
        self.elems = elems
        self.comps = tuple(counts.astype("int").tolist())

    def get_validity(self):
        self.comp_valid = smact_validity(self.elems, self.comps)
        if self.constructed:
            self.struct_valid = structure_validity(self.structure)
        else:
            self.struct_valid = False
        self.valid = self.comp_valid and self.struct_valid


def get_crystal_array_list(file_path: str, multi_eval: bool):
    """
    与 xrd2struc/scripts/compute.py 保持一致。
    multi_eval=False 时，取 batch_idx=0
    multi_eval=True  时，取全部 eval/sample
    """
    data = torch.load(file_path, map_location="cpu")

    if multi_eval:
        batch_size = data["frac_coords"].shape[0]
        crys_array_list = []
        for i in range(batch_size):
            tmp_crys_array_list = get_crystals_list(
                data["frac_coords"][i],
                data["atom_types"][i],
                data["lengths"][i],
                data["angles"][i],
                data["num_atoms"][i],
            )
            crys_array_list.append(tmp_crys_array_list)
    else:
        crys_array_list = get_crystals_list(
            data["frac_coords"][0],
            data["atom_types"][0],
            data["lengths"][0],
            data["angles"][0],
            data["num_atoms"][0],
        )

    if "input_data_batch" in data:
        batch = data["input_data_batch"]
        if isinstance(batch, dict):
            true_crystal_array_list = get_crystals_list(
                batch["frac_coords"],
                batch["atom_types"],
                batch["lengths"],
                batch["angles"],
                batch["num_atoms"],
            )
        else:
            true_crystal_array_list = get_crystals_list(
                batch.frac_coords,
                batch.atom_types,
                batch.lengths,
                batch.angles,
                batch.num_atoms,
            )
    else:
        true_crystal_array_list = None

    return crys_array_list, true_crystal_array_list


# =========================
# 2. 结构匹配与 RMSE
# =========================
def get_matcher():
    return StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)


def get_rms_dist_safe(
    struct_pred: Structure,
    struct_gt: Structure,
    matcher: StructureMatcher,
) -> Optional[float]:
    try:
        rms_dist = matcher.get_rms_dist(struct_gt, struct_pred)
        return None if rms_dist is None else float(rms_dist[0])
    except Exception:
        return None


# =========================
# 3. CIF / GSAS-II / Rietveld
# =========================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_structure_to_cif(structure: Structure, cif_path: str, symprec: float = 0.5):
    CifWriter(structure, symprec=symprec).write_file(cif_path)


def generate_xrd_from_target_cif(
    target_cif: str,
    prmfile: str,
    save_dir: str,
    start: float = 5.0,
    end: float = 80.0,
    gap: float = 0.01,
    norm: bool = False,
) -> str:
    """
    使用 target cif 生成模拟 PXRD，和论文附带脚本保持一致。
    """
    gpx_path = os.path.join(save_dir, "xrd.gpx")
    xrd_path = os.path.join(save_dir, "XRD_tar.dat")

    gpx = G2sc.G2Project(filename=gpx_path)
    gpx.add_phase(target_cif, fmthint="CIF", phasename="B")
    gpx.add_simulated_powder_histogram(
        "A simulation",
        prmfile,
        start,
        end,
        gap,
        phases=gpx.phases(),
        scale=1.0,
    )
    gpx.data["Controls"]["data"]["max cyc"] = 0
    gpx.do_refinements([{}])

    y = np.array(gpx.histogram(0).getdata("ycalc"))
    if norm and np.max(y) > 0:
        y = y / np.max(y)

    intensity = y[:-1]
    thetatwo = np.arange(start, end, gap)
    new_xrd_file = np.stack((thetatwo, intensity)).T
    np.savetxt(xrd_path, new_xrd_file, fmt="%.2f %.7f")

    return xrd_path


def hist_stats_callback(gpx):
    gpx.save()


def rietveld_wo_spacegroup(
    pre_cif: str,
    xrd_dat: str,
    prmfile: str,
    save_dir: str,
) -> Tuple[Optional[float], Optional[str]]:
    """
    第一阶段 refinement：
    直接用预测结构 pre.cif 作为初始结构
    输出 refine.cif
    """
    try:
        gpx = G2sc.G2Project(filename=os.path.join(save_dir, "start.gpx"))
        hist1 = gpx.add_powder_histogram(xrd_dat, prmfile)
        phase0 = gpx.add_phase(pre_cif, phasename="a", histograms=[hist1], fmthint="CIF")
        gpx.data["Controls"]["data"]["max cyc"] = 100

        refdict0 = {"clear": {"Sample Parameters": ["Scale"]}}
        refdict1 = {"set": {"Cell": True}}
        refdict2 = {"set": {"Sample Parameters": ["Scale"]}}
        refdict3 = {
            "set": {"Atoms": {"all": "X"}},
            "histograms": [hist1],
            "output": os.path.join(save_dir, "last_stage1.gpx"),
            "call": hist_stats_callback,
        }

        dict_list = [refdict0, refdict1, refdict2, refdict3]
        gpx.do_refinements(dict_list)

        rwp = None
        for hist in gpx.histograms():
            rwp = hist.get_wR()

        refine_cif = os.path.join(save_dir, "refine.cif")
        phase0.export_CIF(refine_cif)

        return (None if rwp is None else float(rwp), refine_cif)
    except Exception:
        return None, None


def convert_to_conventional_structure(
    in_cif: str,
    out_cif: str,
    symprec: float = 0.5,
) -> bool:
    """
    与作者代码保持一致：先把第一阶段 refinement 结果转成 conventional standard structure
    再作为第二阶段 refinement 输入
    """
    try:
        primitive_one = Structure.from_file(in_cif)
        convention_one = SpacegroupAnalyzer(primitive_one).get_conventional_standard_structure()
        CifWriter(convention_one, symprec=symprec).write_file(out_cif)
        return True
    except Exception:
        return False


def rietveld_constrained_with_spacegroup(
    pre_conventional_cif: str,
    xrd_dat: str,
    prmfile: str,
    save_dir: str,
) -> Tuple[Optional[float], Optional[str]]:
    """
    第二阶段 refinement：
    输入是 conventional structure
    输出 refine_con_last.cif
    """
    try:
        gpx = G2sc.G2Project(filename=os.path.join(save_dir, "start_stage2.gpx"))
        hist1 = gpx.add_powder_histogram(xrd_dat, prmfile)
        phase0 = gpx.add_phase(
            pre_conventional_cif,
            phasename="a",
            histograms=[hist1],
            fmthint="CIF",
        )
        gpx.data["Controls"]["data"]["max cyc"] = 100

        refdict0 = {"clear": {"Sample Parameters": ["Scale"]}}
        refdict1 = {"set": {"Cell": True}}
        refdict2 = {"set": {"Sample Parameters": ["Scale"]}}
        refdict3 = {
            "set": {"Atoms": {"all": "X"}},
            "histograms": [hist1],
            "output": os.path.join(save_dir, "last_stage2.gpx"),
            "call": hist_stats_callback,
        }

        dict_list = [refdict0, refdict1, refdict2, refdict3]
        gpx.do_refinements(dict_list)

        rwp = None
        for hist in gpx.histograms():
            rwp = hist.get_wR()

        refine_con_last_cif = os.path.join(save_dir, "refine_con_last.cif")
        phase0.export_CIF(refine_con_last_cif)

        return (None if rwp is None else float(rwp), refine_con_last_cif)
    except Exception:
        return None, None


# =========================
# 4. 从多采样结果中选择一个候选结构
# =========================
def choose_best_prediction_for_one_material(
    pred_candidates: List[Crystal],
    gt_crystal: Crystal,
    matcher: StructureMatcher,
) -> Tuple[Optional[int], Optional[float], Optional[Crystal]]:
    """
    默认采用 oracle 方式：从多个 sample 中选与 gt RMS 最小的候选。
    这是为了尽量对齐论文 Figure 5a 的 20-sample benchmark 复现。
    """
    best_idx = None
    best_rms = None
    best_crys = None

    for idx, pred in enumerate(pred_candidates):
        if not pred.valid or not gt_crystal.valid:
            continue
        rms = get_rms_dist_safe(pred.structure, gt_crystal.structure, matcher)
        if rms is None:
            continue
        if best_rms is None or rms < best_rms:
            best_rms = rms
            best_idx = idx
            best_crys = pred

    return best_idx, best_rms, best_crys


def build_crystal_objects_from_pt(
    recon_file_path: str,
    multi_eval: bool,
) -> Tuple[List[Crystal], List[List[Crystal]]]:
    """
    返回：
    gt_crys: List[Crystal], 长度 N
    pred_crys_all:
        - multi_eval=False: 只有一个列表 [preds]
        - multi_eval=True:  长度 = num_eval，每个元素是长度 N 的预测 Crystal 列表
    """
    crys_array_list, true_crystal_array_list = get_crystal_array_list(
        recon_file_path, multi_eval=multi_eval
    )

    gt_crys = [Crystal(x) for x in true_crystal_array_list]

    if multi_eval:
        pred_crys_all = []
        for i in range(len(crys_array_list)):
            preds_i = [Crystal(x) for x in crys_array_list[i]]
            pred_crys_all.append(preds_i)
    else:
        pred_crys_all = [[Crystal(x) for x in crys_array_list]]

    return gt_crys, pred_crys_all


# =========================
# 5. 单个样本的完整流程
# =========================
def run_one_material_end2end(
    material_idx: int,
    gt_structure: Structure,
    pred_structure: Structure,
    out_root: str,
    prmfile: str,
    matcher: StructureMatcher,
) -> Dict:
    """
    对单个材料：
    1) 写 target/pred cif
    2) 算 rms_before
    3) 用 target cif 生成模拟 XRD
    4) 两阶段 Rietveld refinement
    5) 算 rms_after
    6) 返回结果 dict
    """
    sample_name = f"sample_{material_idx:05d}"
    sample_dir = os.path.join(out_root, sample_name)
    ensure_dir(sample_dir)

    tar_cif = os.path.join(sample_dir, f"{sample_name}_tar.cif")
    pre_cif = os.path.join(sample_dir, f"{sample_name}_pre.cif")
    refine_con_cif = os.path.join(sample_dir, f"{sample_name}_refine_con.cif")

    result = {
        "material_idx": material_idx,
        "sample_name": sample_name,
        "rms_before": None,
        "rms_after": None,
        "rwp_stage1": None,
        "rwp_stage2": None,
        "refine_success": 0,
        "error_msg": "",
    }

    try:
        write_structure_to_cif(gt_structure, tar_cif, symprec=0.5)
        write_structure_to_cif(pred_structure, pre_cif, symprec=0.5)

        rms_before = get_rms_dist_safe(pred_structure, gt_structure, matcher)
        result["rms_before"] = rms_before

        xrd_dat = generate_xrd_from_target_cif(
            target_cif=tar_cif,
            prmfile=prmfile,
            save_dir=sample_dir,
            start=5.0,
            end=80.0,
            gap=0.01,
            norm=False,
        )

        rwp1, refine_cif = rietveld_wo_spacegroup(
            pre_cif=pre_cif,
            xrd_dat=xrd_dat,
            prmfile=prmfile,
            save_dir=sample_dir,
        )
        result["rwp_stage1"] = rwp1

        if refine_cif is None or (not os.path.exists(refine_cif)):
            result["error_msg"] = "stage1_refinement_failed"
            return result

        ok = convert_to_conventional_structure(
            in_cif=refine_cif,
            out_cif=refine_con_cif,
            symprec=0.5,
        )
        if not ok or (not os.path.exists(refine_con_cif)):
            result["error_msg"] = "convert_to_conventional_failed"
            return result

        rwp2, refine_con_last_cif = rietveld_constrained_with_spacegroup(
            pre_conventional_cif=refine_con_cif,
            xrd_dat=xrd_dat,
            prmfile=prmfile,
            save_dir=sample_dir,
        )
        result["rwp_stage2"] = rwp2

        if refine_con_last_cif is None or (not os.path.exists(refine_con_last_cif)):
            result["error_msg"] = "stage2_refinement_failed"
            return result

        refined_structure = Structure.from_file(refine_con_last_cif)
        rms_after = get_rms_dist_safe(refined_structure, gt_structure, matcher)
        result["rms_after"] = rms_after

        if rms_after is not None:
            result["refine_success"] = 1
        else:
            result["error_msg"] = "rms_after_is_none"

        return result

    except Exception as e:
        result["error_msg"] = f"{type(e).__name__}: {str(e)}"
        return result


# =========================
# 6. 画 Figure 5a
# =========================
def plot_fig5a(
    csv_path: str,
    fig_path: str,
    title: str = "Figure 5a Reproduction",
):
    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    before = []
    after = []

    for row in rows:
        if str(row["refine_success"]) != "1":
            continue

        rb = row["rms_before"]
        ra = row["rms_after"]
        if rb in ("", "None", None) or ra in ("", "None", None):
            continue

        rb = float(rb)
        ra = float(ra)

        if rb > 0 and ra > 0 and np.isfinite(rb) and np.isfinite(ra):
            before.append(rb)
            after.append(ra)

    before = np.array(before, dtype=float)
    after = np.array(after, dtype=float)

    before_sorted = np.sort(before)
    after_sorted = np.sort(after)
    n = min(len(before_sorted), len(after_sorted))
    before_sorted = before_sorted[:n]
    after_sorted = after_sorted[:n]

    x = np.arange(1, n + 1)

    plt.figure(figsize=(7.2, 5.4))
    plt.plot(x, np.log10(after_sorted), label="After Rietveld", linewidth=2.2)
    plt.plot(x, np.log10(before_sorted), label="Before Rietveld", linewidth=2.2)

    plt.xlabel("Counts", fontsize=14)
    plt.ylabel("log(RMSE)", fontsize=14)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()


# =========================
# 7. 主流程
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pt_path",
        type=str,
        required=True,
        help="xrd2struc 的预测结果 .pt 文件路径",
    )
    parser.add_argument(
        "--prmfile",
        type=str,
        required=True,
        help="GSAS-II 使用的 INST_XRY_Cu.PRM 路径",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="输出目录：保存 cif / csv / figure",
    )
    parser.add_argument(
        "--multi_eval",
        action="store_true",
        help="若 .pt 内含多次采样结果，则打开此项（例如 20-sample）",
    )
    parser.add_argument(
        "--max_materials",
        type=int,
        default=-1,
        help="调试用；-1 表示跑全部样本",
    )
    parser.add_argument(
        "--clean_sample_dir",
        action="store_true",
        help="若存在旧 sample_xxxxx 目录，先删除再重建",
    )
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    matcher = get_matcher()

    print("=" * 80)
    print("Loading PT file...")
    print(args.pt_path)
    gt_crys, pred_crys_all = build_crystal_objects_from_pt(
        recon_file_path=args.pt_path,
        multi_eval=args.multi_eval,
    )

    num_materials = len(gt_crys)
    num_evals = len(pred_crys_all)
    print(f"num_materials = {num_materials}")
    print(f"num_evals     = {num_evals}")
    print("=" * 80)

    # 先为每个材料选一个候选预测结构
    selected = []
    print("Selecting one predicted structure for each material...")

    for i in tqdm(range(num_materials)):
        if args.max_materials > 0 and i >= args.max_materials:
            break

        gt_i = gt_crys[i]

        pred_candidates_i = [pred_crys_all[k][i] for k in range(num_evals)]
        best_idx, best_rms, best_crys = choose_best_prediction_for_one_material(
            pred_candidates=pred_candidates_i,
            gt_crystal=gt_i,
            matcher=matcher,
        )

        selected.append(
            {
                "material_idx": i,
                "best_eval_idx": best_idx,
                "oracle_rms_before": best_rms,
                "gt_crystal": gt_i,
                "pred_crystal": best_crys,
            }
        )

    # 跑 Rietveld
    result_csv = os.path.join(args.out_dir, "fig5a_results.csv")
    fig_path = os.path.join(args.out_dir, "fig5a.png")

    fieldnames = [
        "material_idx",
        "sample_name",
        "best_eval_idx",
        "oracle_rms_before",
        "rms_before",
        "rms_after",
        "rwp_stage1",
        "rwp_stage2",
        "refine_success",
        "error_msg",
    ]

    print("=" * 80)
    print("Running end-to-end Rietveld refinement...")
    print("=" * 80)

    with open(result_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for item in tqdm(selected):
            material_idx = item["material_idx"]
            gt_i = item["gt_crystal"]
            pred_i = item["pred_crystal"]

            row = {
                "material_idx": material_idx,
                "sample_name": f"sample_{material_idx:05d}",
                "best_eval_idx": item["best_eval_idx"],
                "oracle_rms_before": item["oracle_rms_before"],
                "rms_before": None,
                "rms_after": None,
                "rwp_stage1": None,
                "rwp_stage2": None,
                "refine_success": 0,
                "error_msg": "",
            }

            if gt_i is None or pred_i is None:
                row["error_msg"] = "no_valid_prediction_selected"
                writer.writerow(row)
                continue

            sample_dir = os.path.join(args.out_dir, f"sample_{material_idx:05d}")
            if args.clean_sample_dir and os.path.isdir(sample_dir):
                shutil.rmtree(sample_dir, ignore_errors=True)

            one_result = run_one_material_end2end(
                material_idx=material_idx,
                gt_structure=gt_i.structure,
                pred_structure=pred_i.structure,
                out_root=args.out_dir,
                prmfile=args.prmfile,
                matcher=matcher,
            )

            row.update(one_result)
            writer.writerow(row)

    print("=" * 80)
    print("Plotting Figure 5a ...")
    plot_fig5a(
        csv_path=result_csv,
        fig_path=fig_path,
        title="Figure 5a reproduction",
    )

    # 打印简单统计
    rows = []
    with open(result_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    n_total = len(rows)
    n_refined = sum(int(r["refine_success"]) == 1 for r in rows)
    n_before_valid = sum(
        r["rms_before"] not in ("", "None", None) for r in rows
    )
    n_after_valid = sum(
        r["rms_after"] not in ("", "None", None) for r in rows
    )

    print("=" * 80)
    print(f"Total selected materials      : {n_total}")
    print(f"Valid before-RMS materials    : {n_before_valid}")
    print(f"Successfully refined materials: {n_refined}")
    print(f"Valid after-RMS materials     : {n_after_valid}")
    print(f"CSV saved to                  : {result_csv}")
    print(f"Figure saved to               : {fig_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

"""
python /workspace/PXRDgen/capsule/code/Rietveld/run_fig5a_end2end.py \
  --pt_path /workspace/mp-20/outs/xrd2struc/flow_CNN_L/last_sample20_0.pt \
  --prmfile /workspace/PXRDgen/capsule/code/Rietveld/INST_XRY_Cu.PRM \
  --out_dir /workspace/mp-20/outs/fig5a_rr \
  --multi_eval
"""