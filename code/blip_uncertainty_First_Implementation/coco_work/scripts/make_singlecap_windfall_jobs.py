from pathlib import Path

configs = [
    ("generatedK3_clean",      "coco_work/subsets/coco_train_500_generatedK3_clean_seed123.json"),
    ("generatedK3_drop40_fix", "coco_work/subsets/coco_train_500_generatedK3_drop40_seed123.json"),
    ("generatedK3_drop60_fix", "coco_work/subsets/coco_train_500_generatedK3_drop60_seed123.json"),
    ("generatedK3_shuffle40",  "coco_work/subsets/coco_train_500_generatedK3_shuffle40_seed123.json"),
    ("generatedK3_shuffle60",  "coco_work/subsets/coco_train_500_generatedK3_shuffle60_seed123.json"),
    ("generatedK3_mismatch40", "coco_work/subsets/coco_train_500_generatedK3_mismatch40_seed123.json"),
    ("generatedK3_mismatch60", "coco_work/subsets/coco_train_500_generatedK3_mismatch60_seed123.json"),
]

out_dir = Path("job_scripts_singlecap")
out_dir.mkdir(exist_ok=True)

for tag, input_path in configs:
    script_path = out_dir / f"run_singlecap_{tag}.sh"
    script = f"""#!/bin/bash
#SBATCH --job-name=sc_{tag}
#SBATCH --account=windfall
#SBATCH --partition=gpu_windfall
#SBATCH --qos=part_qos_windfall
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=./blip_uncertainty/logs/singlecap_{tag}_%j.out

module purge
module load pytorch/nvidia/22.12
cd ./blip_uncertainty
export PYTHONUNBUFFERED=1

pytorch coco_work/scripts/train_single_caption_cli.py \\
  --run_tag singlecap_{tag}_seed123 \\
  --input_path {input_path} \\
  --max_images 500 \\
  --epochs 10
"""
    script_path.write_text(script)
    print("Saved:", script_path)
