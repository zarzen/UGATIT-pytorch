#!/bin/bash

WORLD_SIZE=4
ASYNC=1
OP=allgather
END_RANK=$(expr ${WORLD_SIZE} - 1)

echo WORLD_SIZE=${WORLD_SIZE}
echo ASYNC=$ASYNC
echo OP=${OP}


for i in $(seq 0 ${END_RANK})
do
if [ $i -eq ${END_RANK} ]
then 
  CUDA_VISIBLE_DEVICES=${i} python profile_collops.py --world-size ${WORLD_SIZE} --rank ${i} --op ${OP} --asyn=$ASYNC
else
  CUDA_VISIBLE_DEVICES=${i} python profile_collops.py --world-size ${WORLD_SIZE} --rank ${i} --op ${OP} --asyn=$ASYNC &
fi
done
