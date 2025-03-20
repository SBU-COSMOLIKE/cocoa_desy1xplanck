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

#SBATCH --time=120:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jiachuanxu@arizona.edu

# Clear the environment from any previously loaded modules
module purge > /dev/null 2>&1
module load anaconda
module load openmpi5
conda init bash
source ~/.bashrc 

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job NAME is $SLURM_JOB_NAME
echo Slurm job ID is $SLURM_JOBID
echo Loaded modules: $(module list)
#echo Available Ethernet: $(ifconfig)

###SAMPLEEMU=./projects/desy1xplanck/emulator/sample_emulator.py
YAML_EMU="./projects/desy1xplanck/yaml/Y3xPLKR4_EMU_science/EmuMCMC_science_${SLURM_ARRAY_TASK_ID}.yaml"
echo Enter working dir ${SLURM_SUBMIT_DIR}
cd $SLURM_SUBMIT_DIR
echo Activating conda and cocoa environment
conda activate cocoatorch
source start_cocoa
#export OMPI_MCA_oob_tcp_if_include=all
#export OMPI_MCA_btl_tcp_if_include=all
export OMP_PROC_BIND=close
if [ -n "$SLURM_CPUS_PER_TASK" ]; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
else
  export OMP_NUM_THREADS=1
fi

echo Using mpirun from $(which mpirun)
echo Slurm job ntask is $SLURM_NTASKS
echo Slurm job nthread is $OMP_NUM_THREADS
echo Running job ${YAML_EMU}
MPI_PML="--mca pml ob1"
MPI_VERBOSE="--mca btl_base_verbose 100"
MPI_HWLOC="--mca hwloc_base_binding_policy none"
MPI_ORTE="--mca orte_base_help_aggregate 0"
MPI_BTL="--mca btl tcp,self"
MPI_BIND="--oversubscribe --bind-to core --map-by node:pe=${OMP_NUM_THREADS}"

mpirun -n ${SLURM_NTASKS} ${MPI_PML} ${MPI_BTL} cobaya-run ${YAML_EMU} -r

