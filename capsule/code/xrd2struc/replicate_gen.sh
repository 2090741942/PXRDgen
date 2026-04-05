python /workspace/PXRDgen/capsule/code/xrd2struc/scripts/evaluate_diffusion.py \
  --model_path /workspace/PXRDgen/capsule/code/xrd2struc/outputs/2026-03-25/Diff_CNN_1.00_unfixed --num_evals 1 --label -1 --order 1

python /workspace/PXRDgen/capsule/code/xrd2struc/scripts/evaluate_diffusion.py \
  --model_path /workspace/PXRDgen/capsule/code/xrd2struc/outputs/2026-03-25/Diff_CNN_1.00_unfixed --num_evals 1 --label -1 --order 2

python scripts/compute.py \
  --pt_path /workspace/PXRDgen/capsule/code/xrd2struc/outputs/2026-03-25/Diff_CNN_1.00_unfixed/last_sample1_1.pt >> z_train.txt


python scripts/compute.py \
  --pt_path /workspace/PXRDgen/capsule/code/xrd2struc/outputs/2026-03-25/Diff_CNN_1.00_unfixed/last_sample1_2.pt >> z_train.txt














