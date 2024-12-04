#!/bin/bash

### Train emulator with ResNet model (Evan's version, no TRF, 1 iter)

#SBATCH --job-name=EMUTRES
#SBATCH --output=/xdisk/timeifler/jiachuanxu/job_logs/emutrain_resnet.out

#SBATCH --account=timeifler
#SBATCH --partition=gpu_standard
#SBATCH --qos=qual_qos_timeifler
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=28
#SBATCH --mem-per-cpu=6gb

#SBATCH --time=96:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jiachuanxu@arizona.edu

# Clear the environment from any previously loaded modules
module purge > /dev/null 2>&1
module load openmpi3
conda init bash
source ~/.bashrc

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job NAME is $SLURM_JOB_NAME
echo Slurm job ID is $SLURM_JOBID

cd /home/u17/jiachuanxu/cocoa/Cocoa/
conda activate cocoatorch
source start_cocoa

MPIRUN=/groups/timeifler/jiachuanxu/python_envs/envs/cocoatorch/bin/mpirun

CONFIGFILE=./projects/desy1xplanck/emulator_deprecated/train_emu_full_resnet.yaml
TRAINEMU=./projects/desy1xplanck/emulator_deprecated/train_emulator.py
SAMPLEEMU=./projects/desy1xplanck/emulator_deprecated/sample_emulator.py

export OMP_NUM_THREADS=1

$PYTHON3 -u ${TRAINEMU} $CONFIGFILE

#mpirun -n ${SLURM_NTASKS} --mca btl tcp,self ${PYTHON3} -u ${SAMPLEEMU} ${CONFIGFILE} -r

