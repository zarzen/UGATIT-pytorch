#!/bin/bash

WORLD_SIZE=4

END_RANK=$(expr ${WORLD_SIZE} - 1)

echo WORLD_SIZE=${WORLD_SIZE}

for i in $(seq 0 ${END_RANK})
do
if [ $i -eq ${END_RANK} ]
then 
  CUDA_VISIBLE_DEVICES=${i} python test_mp_gen.py --world-size ${WORLD_SIZE} --rank ${i}
else
  CUDA_VISIBLE_DEVICES=${i} python test_mp_gen.py --world-size ${WORLD_SIZE} --rank ${i} &
fi
done