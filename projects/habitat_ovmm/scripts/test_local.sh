#!/usr/bin/env bash

DOCKER_NAME="ovmm_baseline_submission"

while [[ $# -gt 0 ]]
do
key="${1}"

case $key in
      --docker-name)
      shift
      DOCKER_NAME="${1}"
	  shift
      ;;
    *)
      echo unkown arg ${1}
      exit
      ;;
esac
done

docker run \
      -v $(pwd)/habitat-challenge-data:/habitat-challenge-data \
      -v $(realpath data):/home-robot/data \
      --runtime=nvidia \
      -e "AGENT_EVALUATION_TYPE=local" \
      ${DOCKER_NAME}
