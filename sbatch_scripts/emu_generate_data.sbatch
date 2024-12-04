#!/bin/bash

### Calculate training and validation dataset using cocoa

#SBATCH --job-name=EMUGenData
#SBATCH --output=/xdisk/timeifler/jiachuanxu/job_logs/emu_gen_data.out

#SBATCH --nodes=12
#SBATCH --ntasks-per-node=28
#SBATCH --exclusive

##SBATCH --partition=high_priority
##SBATCH --qos=user_qos_timeifler
#SBATCH --partition=standard
#SBATCH --qos=qual_qos_timeifler

#SBATCH --account=timeifler
#SBATCH --time=120:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jiachuanxu@arizona.edu

# Clear the environment from any previously loaded modules
module purge > /dev/null 2>&1
module load openmpi3
conda init bash
source ~/.bashrc

echo Running on host `hostname`
###$PYTHON3 ${TRAINEMU} $CONFIGFILE 0 --temper --save_emu --debug
echo Time is `date`
echo Directory is `pwd`
echo Slurm job NAME is $SLURM_JOB_NAME
echo Slurm job ID is $SLURM_JOBID

cd /home/u17/jiachuanxu/cocoa/Cocoa/
conda activate cocoatorch
source start_cocoa

MPIRUN=/groups/timeifler/jiachuanxu/python_envs/envs/cocoatorch/bin/mpirun

CONFIGFILE=./projects/desy1xplanck/emulator_deprecated/train_emu_full_resnet.yaml
CALDV=./projects/desy1xplanck/emulator_deprecated/calculate_dv.py

export OMP_NUM_THREADS=1

${MPIRUN} -n ${SLURM_NTASKS} --mca btl tcp,self $PYTHON3 ${CALDV} $CONFIGFILE
