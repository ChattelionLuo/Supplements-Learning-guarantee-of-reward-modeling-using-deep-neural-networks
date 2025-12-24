#!/bin/bash
mkdir ~/tmp
export TMPDIR=~/tmp

Target_file="sim2.py"
njob=200

ncores=1
export OMP_NUM_THREADS=$ncores
export OPENBLAS_NUM_THREADS=$ncores
export MKL_NUM_THREADS=$ncores
export NUMEXPR_NUM_THREADS=$ncores
export VECLIB_MAXIMUM_THREADS=$ncores


ccount=0

# Remove the logs directory if it exists and create a new one
if [ -d "logs2" ]; then
    rm -rf logs2
fi
mkdir logs2
 
for iloop in $(seq 1 400)
do

    # iloop=$(($vv + $idx_Begin - 1))
    ccount=$(($ccount+1))
    iflag=$(($ccount % $njob))
    echo -n "icount-"$ccount
    if [ $iflag == 0 ]
    then
    sleep 1s
        nohup  python3 -u $Target_file  --iloop  $iloop  >> "logs2/Zresult"$iloop".log" 2>&1
    else         
        nohup  python3 -u $Target_file  --iloop  $iloop  >> "logs2/Zresult"$iloop".log" 2>&1 &
    fi
done