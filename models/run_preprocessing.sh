#!/bin/bash

if [ -z "$3" ]; then
    echo "Must enter 3 agruments"
    echo "1: Path to Input File (../pythia/output/<file>.root)"
    echo "2: Path to Output File (data/<file>.pkl)"
    echo "3: Path to Output Directory (plots/<Dir Name>) e.g. plots/preprocessing"
    exit 1
fi

in_file=$1
out_file=$2
out_plots=$3

mkdir -p $out_plots
nohup python -u preprocessing.py $run_type $in_file $out_file $out_plots > "${out_plots}/preprocessing.log" 2>&1 &

# To watch the progress of the script, uncomment the following command:
tail -f "${out_plots}/preprocessing.log"
