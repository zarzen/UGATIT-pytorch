#!/bin/bash

WORLD_SIZE=4
REPEAT=300
WARM=200
IMG_SIZE=256
NGF=64
BS=8

END_RANK=$(expr ${WORLD_SIZE} - 1)

echo WORLD_SIZE=${WORLD_SIZE}
echo repeat=${REPEAT} "times"
echo warmup=${WARM} "times"
echo image_size=${IMG_SIZE}
echo ngf=${NGF}
echo batch-size=${BS}

for i in $(seq 0 ${END_RANK})
do
if [ $i -eq ${END_RANK} ]
then 
  CUDA_VISIBLE_DEVICES=${i} python profile_mp_gen.py --world-size ${WORLD_SIZE} --rank ${i} --img-size ${IMG_SIZE} --repeat ${REPEAT} --warm-up ${WARM} --batch-size ${BS} --ngf ${NGF}
else
  CUDA_VISIBLE_DEVICES=${i} python profile_mp_gen.py --world-size ${WORLD_SIZE} --rank ${i} --img-size ${IMG_SIZE} --repeat ${REPEAT} --warm-up ${WARM} --batch-size ${BS} --ngf ${NGF} &
fi
done