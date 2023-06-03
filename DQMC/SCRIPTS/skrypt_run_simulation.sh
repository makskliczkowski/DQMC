#!/bin/bash
#SBATCH -N1
#SBATCH -c1
#SBATCH --mem=1gb
#SBATCH --time=1:00:00
#SBATCH --job-name=vqmc-loop
Lx=$1
Ly=$2
dtau=$3
beta=$4
M0=$5
U=$6
mu=$7
folder=$8

dir="cd /home/klimak97/CODES/DQMC/DQMC/"
echo $dir
cd /home/klimak97/CODES/DQMC/DQMC/


a="L=${Lx}x${Ly}_beta=${beta},mu=${mu},U=${U}"
echo "#!/bin/bash" >> ${a}
echo "#SBATCH -N1" >> ${a}
echo "#SBATCH -c1" >> ${a}
echo "#SBATCH --mem=16gb" >> ${a}
echo "#SBATCH --time=99:59:59" >> ${a}
echo >> ${a}
echo "#SBATCH --job-name=${a}" >> ${a}
echo >> ${a}
echo "source /usr/local/sbin/modules.sh" >> ${a}
echo >> ${a}
echo "module load intel" >> ${a}
echo >> ${a}
echo "module load OpenMPI" >> ${a}
echo >> ${a}
echo "module load HDF5" >> ${a}
echo >> ${a}
echo "cd /home/klimak97/CODES/DQMC/DQMC/" >> ${a}
echo >> ${a}
echo "./dqmc.o -fun 11 -d 2 -Lx ${Lx} -Ly ${Ly} -beta ${beta} -dtau ${dtau} -M0 ${M0} -mu ${mu} -U ${U} -th 1 -dir ${folder} -mcS 1000 -mcA 5000 -mcC 1 >& ./LOG/log ${a}.txt" >> ${a}
sbatch ${a}
rm ${a}

echo "finished"