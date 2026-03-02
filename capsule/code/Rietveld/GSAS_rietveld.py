import sys, os
sys.path.insert(0, r'/g2full/GSAS-II/GSASII') ## remember to change it!
import GSASIIscriptable as G2sc
import numpy as np
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.analysis.structure_matcher import StructureMatcher


G2sc.SetPrintLevel('none')

work_dir = '/code/Rietveld/Ce6Al3Ga3Ni6'
save_dir = '/results'
prmfile = '/code/Rietveld/INST_XRY_Cu.PRM'
name = 'Ce6Al3Ga3Ni6'

def XRD_data(cif_file, start=5.0, end=80.0, gap=0.01, norm=False):
    gpx = G2sc.G2Project(filename=os.path.join(save_dir, 'xrd.gpx') )
    phase = gpx.add_phase(cif_file, fmthint='CIF', phasename='B')
    gpx.add_simulated_powder_histogram('A simulation', prmfile, start, end, gap, phases=gpx.phases(), scale=1.0)
    gpx.data['Controls']['data']['max cyc'] = 0
    gpx.do_refinements([{}])
    y = gpx.histogram(0).getdata('ycalc')
    y = np.array(y)
    if norm:
        y = y/np.max(y)
    
    intensity = y[:-1]
    thetatwo = np.arange(5, 80, 0.01)
    new_xrd_file = np.stack((thetatwo, intensity)).T
    save_xrd = os.path.join(save_dir, 'XRD_tar.dat')
    np.savetxt(save_xrd, new_xrd_file, fmt='%.2f %.7f')
    
    return save_xrd
    

def HistStats(gpx):
    gpx.save()


def Rietveld_wo_spacegroup(name, save_xrd):
    
    gpx = G2sc.G2Project(filename=os.path.join(save_dir, 'start.gpx'))
    hist1 = gpx.add_powder_histogram(save_xrd, prmfile)
    phase0 = gpx.add_phase(os.path.join(work_dir, name+'_pre.cif'), phasename='a',histograms=[hist1],fmthint='CIF')
    gpx.data['Controls']['data']['max cyc'] = 100

    refdict0 = {"clear":{'Sample Parameters':['Scale']}}
    refdict1 = {"set": {"Cell": True}}
    refdict2 = {"set":{'Sample Parameters':['Scale']}}
    refdict3 = {"set": {"Atoms":{"all":"X"}}, "histograms":[hist1], "output":work_dir+'/last.gpx',"call":HistStats}

    dictList = [refdict0,refdict1,refdict2,refdict3]
    gpx.do_refinements(dictList)
    for hist in gpx.histograms():
        _, Rwp = hist.name, hist.get_wR()
    print('--------------------- Rwp --------------')
    print(round(Rwp, 5))
    phase0.export_CIF(os.path.join(save_dir, name+'_refine.cif'))


def Rietveld_constrained_with_spacegroup(name, save_xrd):

    gpx = G2sc.G2Project(filename=os.path.join(save_dir, 'start.gpx'))
    hist1 = gpx.add_powder_histogram(save_xrd, prmfile)
    phase0 = gpx.add_phase(os.path.join(work_dir, name+'_refine_con.cif'), phasename='a',histograms=[hist1],fmthint='CIF')
    gpx.data['Controls']['data']['max cyc'] = 100

    refdict0 = {"clear":{'Sample Parameters':['Scale']}}
    refdict1 = {"set": {"Cell": True}}
    refdict2 = {"set":{'Sample Parameters':['Scale']}}
    refdict3 = {"set": {"Atoms":{"all":"X"}}, "histograms":[hist1], "output":work_dir+'/last.gpx',"call":HistStats}

    dictList = [refdict0,refdict1,refdict2,refdict3]
    gpx.do_refinements(dictList)
    for hist in gpx.histograms():
        _, Rwp = hist.name, hist.get_wR()
    print('--------------------- Rwp --------------')
    print(round(Rwp, 5))
    phase0.export_CIF(os.path.join(save_dir, name+'_refine_con_last.cif'))



save_xrd = XRD_data(os.path.join(work_dir, name+'_tar.cif'))

sm = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)     # True -> False
struct1 = Structure.from_file(os.path.join(work_dir,  name+'_tar.cif'))
struct2 = Structure.from_file(os.path.join(work_dir, name+'_pre.cif'))
rms_dist = sm.get_rms_dist(struct1, struct2)
rms_dist = None if rms_dist is None else rms_dist[0]
print('-----------rms before Rietveld-----------')
print(rms_dist)


Rietveld_wo_spacegroup(name, save_xrd)

primitive_one = Structure.from_file(os.path.join(save_dir, name+'_refine.cif'))
convention_file = os.path.join(save_dir, name+'_refine_con.cif')
convention_one = SpacegroupAnalyzer(primitive_one).get_conventional_standard_structure()
CifWriter(convention_one, symprec=0.5).write_file(convention_file)

Rietveld_constrained_with_spacegroup(name, save_xrd)
struct3 = Structure.from_file(os.path.join(save_dir, name+'_refine_con_last.cif'))
rms_dist = sm.get_rms_dist(struct1, struct3)
rms_dist = None if rms_dist is None else rms_dist[0]
print('-----------rms after Rietveld-----------')
print(rms_dist)
