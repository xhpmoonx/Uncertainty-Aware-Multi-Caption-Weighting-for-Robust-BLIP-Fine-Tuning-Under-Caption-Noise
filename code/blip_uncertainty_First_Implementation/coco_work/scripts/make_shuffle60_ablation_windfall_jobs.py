from pathlib import Path

OUT_DIR = Path("job_scripts_ablation_shuffle60")
OUT_DIR.mkdir(exist_ok=True)

TRAIN_SCRIPT = "coco_work/scripts/train_dynamic_joint_phase2_cli.py"
INPUT_PATH = "coco_work/subsets/coco_train_500_generatedK3_shuffle60_seed123.json"
WORKDIR = "./blip_uncertainty"
LOG_DIR = f"{WORKDIR}/logs"

SBATCH_HEADER = """#!/bin/bash
#SBATCH --account=windfall
#SBATCH --partition=gpu_windfall
#SBATCH --qos=part_qos_windfall
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output={log_path}

module purge
module load pytorch/nvidia/22.12
cd {workdir}
"""

jobs = [
    {
        "name": "ablate_shuffle60_alignonly",
        "run_tag": "ablate_shuffle60_alignonly_seed123",
        "use_alignment": 1,
        "use_fluency": 0,
        "use_agreement": 0,
    },
    {
        "name": "ablate_shuffle60_fluencyonly",
        "run_tag": "ablate_shuffle60_fluencyonly_seed123",
        "use_alignment": 0,
        "use_fluency": 1,
        "use_agreement": 0,
    },
    {
        "name": "ablate_shuffle60_agreementonly",
        "run_tag": "ablate_shuffle60_agreementonly_seed123",
        "use_alignment": 0,
        "use_fluency": 0,
        "use_agreement": 1,
    },
    {
        "name": "ablate_shuffle60_align_fluency",
        "run_tag": "ablate_shuffle60_align_fluency_seed123",
        "use_alignment": 1,
        "use_fluency": 1,
        "use_agreement": 0,
    },
    {
        "name": "ablate_shuffle60_align_agreement",
        "run_tag": "ablate_shuffle60_align_agreement_seed123",
        "use_alignment": 1,
        "use_fluency": 0,
        "use_agreement": 1,
    },
    {
        "name": "ablate_shuffle60_fluency_agreement",
        "run_tag": "ablate_shuffle60_fluency_agreement_seed123",
        "use_alignment": 0,
        "use_fluency": 1,
        "use_agreement": 1,
    },
]

for job in jobs:
    script_path = OUT_DIR / f"{job['name']}.sh"
    log_path = f"{LOG_DIR}/{job['name']}_%j.out"

    header = SBATCH_HEADER.format(
        log_path=log_path,
        workdir=WORKDIR,
    )

    body = f"""pytorch {TRAIN_SCRIPT} \\
  --run_tag {job['run_tag']} \\
  --input_path {INPUT_PATH} \\
  --max_images 500 \\
  --epochs 10 \\
  --seed 123 \\
  --use_alignment {job['use_alignment']} \\
  --use_fluency {job['use_fluency']} \\
  --use_agreement {job['use_agreement']}
"""

    script_path.write_text(header + "\n" + body)
    print(f"Saved: {script_path}")
