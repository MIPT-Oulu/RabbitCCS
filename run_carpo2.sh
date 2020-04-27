#!/bin/bash
#SBATCH --job-name=thickness-uCT
#SBATCH --mail-user=santeri.rytky@oulu.fi
#SBATCH --mail-type=ALL
#SBATCH --partition=medium
#SBATCH -w ca1-12
#SBATCH -o slurm.%j.out # STDOUT
#SBATCH -e slurm.%j.err # STDERR
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8g

env_name='carpo'
env_created=$(conda env list | grep $env_name)
if [[ ! $env_created ]];
then
	eval "$(conda shell.bash hook)"
	conda env create -f requirements.txt
fi
eval "$(conda shell.bash hook)"
module load GCC/8.2.0-2.31.1
module load OpenMPI/3.1.3-GCC-8.2.0-2.31.1
conda activate carpo
echo "Start the job..."
python -m scripts.thickness_analysis_EP
echo "Done the job!"