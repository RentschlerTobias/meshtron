#!/bin/sh

#PBS -l walltime=24:00:00
#PBS -l nodes=mars.ihs.uni-stuttgart.de:ppn=8
# go to submit directory
#PBS -N QuadMeshtron
cd $PBS_O_WORKDIR

source /mnt/scratch/trentschler/environments/python/env_ai/bin/activate
export OSLO_LOCK_PATH=~/tmp

export FOAM_SIGFPE=false

python main.py >output_mars.log

deactivate
