#!/bin/bash
#SBATCH --job-name=glot500_exp
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100
#SBATCH --clusters=wice
#SBATCH --account=intro_vsc37132
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=robin.edu.hr@gmail.com

module purge
module load Python/3.9

export PYTHONPATH=$PYTHONPATH:$PWD
export HF_HOME=$VSC_DATA/hf_cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

bash setup_vsc_environment.sh

if [ -z "$1" ]; then
    bash run_all_experiments.sh
else
    case "$1" in
        "basic")
            bash run_basic_experiments.sh
            ;;
        "control")
            bash run_control_experiments.sh
            ;;
        "submetric")
            bash run_all_submetrics.sh
            ;;
        "cross")
            bash run_all_cross_lingual.sh
            ;;
        *)
            echo "Unknown experiment type: $1"
            echo "Valid options: basic, control, submetric, cross"
            exit 1
            ;;
    esac
fi