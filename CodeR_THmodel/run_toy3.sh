#!/bin/bash
mkdir ~/tmp
export TMPDIR=~/tmp

Target_file="sim3.py"
njob=200

ncores=1
export OMP_NUM_THREADS=$ncores
export OPENBLAS_NUM_THREADS=$ncores
export MKL_NUM_THREADS=$ncores
export NUMEXPR_NUM_THREADS=$ncores
export VECLIB_MAXIMUM_THREADS=$ncores


ccount=0

# Remove the logs directory if it exists and create a new one
if [ -d "logs3" ]; then
    rm -rf logs3
fi
mkdir logs3
 
for iloop in $(seq 1 16000)
do

    # iloop=$(($vv + $idx_Begin - 1))
    ccount=$(($ccount+1))
    iflag=$(($ccount % $njob))
    echo -n "icount-"$ccount
    if [ $iflag == 0 ]
    then
    sleep 1s
        nohup  python3 -u $Target_file  --iloop  $iloop  >> "logs3/Zresult"$iloop".log" 2>&1
    else         
        nohup  python3 -u $Target_file  --iloop  $iloop  >> "logs3/Zresult"$iloop".log" 2>&1 &
    fi
done