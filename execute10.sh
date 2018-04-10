#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=1
#SBATCH --job-name="kmeans"
#SBATCH --output="kmeans_data_n10.out"
module load intel
for i in {1..10}
do
    for k in 2 4 6 8 10 20
    do
        mpirun david_impl/kmeans 1000000 $k y
    done
done
