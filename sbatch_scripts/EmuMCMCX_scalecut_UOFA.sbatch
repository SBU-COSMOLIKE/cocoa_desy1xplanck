#!/bin/bash

#SBATCH --job-name=EmuSamp
#SBATCH --output=/xdisk/timeifler/jiachuanxu/job_logs/Y3R4Emu-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1

### >>> High priority purchase-in time
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_timeifler
### >>> Qualified special project request
###SBATCH --partition=standard
###SBATCH --qos=qual_qos_timeifler

#SBATCH --account=timeifler

#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jiachuanxu@arizona.edu

# Clear the environment from any previously loaded modules
module purge > /dev/null 2>&1
module load anaconda
conda init bash
source ~/.bashrc 

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job NAME is $SLURM_JOB_NAME
echo Slurm job ID is $SLURM_JOBID

YAML_EMU="./projects/desy1xplanck/yaml/Y3xPLKR4_EMU/EmuMCMC_scalecut_v11_${SLURM_ARRAY_TASK_ID}.yaml"
cd $SLURM_SUBMIT_DIR
conda activate cocoatorch
source start_cocoa

export OMP_PROC_BIND=close
if [ -n "$SLURM_CPUS_PER_TASK" ]; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
else
  export OMP_NUM_THREADS=1
fi

echo Slurm job ntask is $SLURM_NTASKS
echo Slurm job nthread is $OMP_NUM_THREADS
echo Running job ${YAML_EMU}

mpirun -n ${SLURM_NTASKS} --mca btl tcp,self --mca pml ob1 cobaya-run ${YAML_EMU} -r

