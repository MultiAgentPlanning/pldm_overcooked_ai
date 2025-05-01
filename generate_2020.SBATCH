#!/bin/bash
#SBATCH --job-name=generate_2020 # Job name
#SBATCH --output=generate_2020.out  # Standard output file
#SBATCH --error=generate_2020.err  # Standard error file
#SBATCH --ntasks=1  # Number of tasks (usually 1 for Python jobs)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1  # Number of CPU cores per task (adjust as needed)
#SBATCH --mem=96GB  # Memory per task (adjust as needed)
#SBATCH --time=30:00  # Maximum runtime (HH:MM:SS)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aj4332@nyu.edu

module purge

if [ -e /dev/nvidia0 ]; then nv="--nv"; fi

# Run the job inside the Singularity container
singularity exec $nv --overlay /scratch/aj4332/rl_env/overlay-50G-10M.ext3:ro \
  /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash -c \
  "echo 'Sourcing env'; source /ext3/env.sh; echo 'Running script'; python generate_2020_samples.py"