#!/bin/bash

N=30
POLICIES=(baseline double biased)
BASE_DIR="output"

RECOMPILE_DELAY=8
RERUN_DELAY=2

mkdir output

for policy in ${POLICIES[@]}; do
  git switch $policy

  make clean > /dev/null

  echo "Recompiling for policy $policy"
  make qemu-nox < <(sleep $RECOMPILE_DELAY; echo "shutdown") 2&>1 > /dev/null

  echo "Priming with throwaway run"
  make qemu-nox < <(sleep $RERUN_DELAY; echo "test") > /dev/null

  for ((k=1; k<=N; k++)); do
    filename="$BASE_DIR/$policy-$k.txt"
    echo $filename

    make qemu-nox < <(sleep $RERUN_DELAY; echo "test") > $filename

  done
done
