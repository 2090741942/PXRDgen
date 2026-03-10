import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/workspace/g2full/GSAS-II")
from GSASII import GSASIIscriptable as G2sc

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

G2sc.SetPrintLevel("none")


def safe_rms(sm, s1, s2):
    try:
        rms = sm.get_rms_dist(s1, s2)
        return None if rms is None else float(rms[0])
    except Exception:
        return None


def generate_xrd_from_target_cif(tar_cif, prmfile, work_dir, start=5.0, end=80.0, gap=0.01):
    gpx_path = os.path.join(work_dir, "xrd.gpx")
    xrd_path = os.path.join(work_dir, "XRD_tar.dat")

    gpx = G2sc.G2Project(filename=gpx_path)
    gpx.add_phase(tar_cif, fmthint="CIF", phasename="B")
    gpx.add_simulated_powder_histogram(
        "A simulation", prmfile, start, end, gap, phases=gpx.phases(), scale=1.0
    )
    gpx.data["Controls"]["data"]["max cyc"] = 0
    gpx.do_refinements([{}])

    y = np.array(gpx.histogram(0).getdata("ycalc"))
    intensity = y[:-1]
    thetatwo = np.arange(start, end, gap)

    new_xrd_file = np.stack((thetatwo, intensity)).T
    np.savetxt(xrd_path, new_xrd_file, fmt="%.2f %.7f")
    return xrd_path


def hist_stats(gpx):
    gpx.save()


def rietveld_stage1(pre_cif, xrd_dat, prmfile, work_dir):
    gpx = G2sc.G2Project(filename=os.path.join(work_dir, "start_stage1.gpx"))
    hist1 = gpx.add_powder_histogram(xrd_dat, prmfile)
    phase0 = gpx.add_phase(pre_cif, phasename="a", histograms=[hist1], fmthint="CIF")
    gpx.data["Controls"]["data"]["max cyc"] = 100

    refdict0 = {"clear": {"Sample Parameters": ["Scale"]}}
    refdict1 = {"set": {"Cell": True}}
    refdict2 = {"set": {"Sample Parameters": ["Scale"]}}
    refdict3 = {
        "set": {"Atoms": {"all": "X"}},
        "histograms": [hist1],
        "output": os.path.join(work_dir, "last_stage1.gpx"),
        "call": hist_stats,
    }

    gpx.do_refinements([refdict0, refdict1, refdict2, refdict3])

    rwp = None
    for hist in gpx.histograms():
        rwp = hist.get_wR()

    refine_cif = os.path.join(work_dir, "refine.cif")
    phase0.export_CIF(refine_cif)
    return rwp, refine_cif


def convert_to_conventional(in_cif, out_cif, symprec=0.5):
    primitive_one = Structure.from_file(in_cif)
    convention_one = SpacegroupAnalyzer(primitive_one).get_conventional_standard_structure()
    CifWriter(convention_one, symprec=symprec).write_file(out_cif)


def rietveld_stage2(refine_con_cif, xrd_dat, prmfile, work_dir):
    gpx = G2sc.G2Project(filename=os.path.join(work_dir, "start_stage2.gpx"))
    hist1 = gpx.add_powder_histogram(xrd_dat, prmfile)
    phase0 = gpx.add_phase(refine_con_cif, phasename="a", histograms=[hist1], fmthint="CIF")
    gpx.data["Controls"]["data"]["max cyc"] = 100

    refdict0 = {"clear": {"Sample Parameters": ["Scale"]}}
    refdict1 = {"set": {"Cell": True}}
    refdict2 = {"set": {"Sample Parameters": ["Scale"]}}
    refdict3 = {
        "set": {"Atoms": {"all": "X"}},
        "histograms": [hist1],
        "output": os.path.join(work_dir, "last_stage2.gpx"),
        "call": hist_stats,
    }

    gpx.do_refinements([refdict0, refdict1, refdict2, refdict3])

    rwp = None
    for hist in gpx.histograms():
        rwp = hist.get_wR()

    refine_con_last_cif = os.path.join(work_dir, "refine_con_last.cif")
    phase0.export_CIF(refine_con_last_cif)
    return rwp, refine_con_last_cif


def run_one(row, prmfile):
    work_dir = row["work_dir"]
    pre_cif = row["pre_cif"]
    tar_cif = row["tar_cif"]

    sm = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)

    struct_tar = Structure.from_file(tar_cif)
    struct_pre = Structure.from_file(pre_cif)

    rms_before = safe_rms(sm, struct_tar, struct_pre)

    xrd_dat = generate_xrd_from_target_cif(tar_cif, prmfile, work_dir)

    rwp1, refine_cif = rietveld_stage1(pre_cif, xrd_dat, prmfile, work_dir)

    refine_con_cif = os.path.join(work_dir, "refine_con.cif")
    convert_to_conventional(refine_cif, refine_con_cif)

    rwp2, refine_con_last_cif = rietveld_stage2(refine_con_cif, xrd_dat, prmfile, work_dir)

    struct_refined = Structure.from_file(refine_con_last_cif)
    rms_after = safe_rms(sm, struct_tar, struct_refined)

    return rms_before, rms_after, rwp1, rwp2


def main(args):
    df = pd.read_csv(args.manifest)
    results = []

    for _, row in df.iterrows():
        rms_before = None
        rms_after = None
        rwp1 = None
        rwp2 = None
        refine_success = 0
        error_msg = ""

        try:
            rms_before, rms_after, rwp1, rwp2 = run_one(row, args.prmfile)
            if rms_after is not None:
                refine_success = 1
        except Exception as e:
            error_msg = str(e)

        results.append({
            "mat_id": row["mat_id"],
            "best_eval_idx": row.get("best_eval_idx", None),
            "oracle_rms_before": row.get("oracle_rms_before", None),
            "rms_before": rms_before,
            "rms_after": rms_after,
            "rwp_stage1": rwp1,
            "rwp_stage2": rwp2,
            "refine_success": refine_success,
            "error_msg": error_msg,
        })

    save_path = os.path.join(args.out_dir, "fig5a_results.csv")
    pd.DataFrame(results).to_csv(save_path, index=False)
    print(f"results saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prmfile", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    main(args)

"""
bash run_fig5a_pipeline.sh \
/workspace/mp-20/outs/xrd2struc/flow_CNN_L/last_sample20_0.pt \
/workspace/PXRDgen/capsule/code/Rietveld/INST_XRY_Cu.PRM \
/workspace/mp-20/outs/fig5a_rr \
--multi_eval
"""