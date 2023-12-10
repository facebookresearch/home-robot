#!/bin/bash
echo "number of episodes per job: 100"
for START in {0..1150..50}
do
    echo "start episode is: $START"
    sbatch --job-name "eplan..$START" ./data_collection_launch.sh $START
done