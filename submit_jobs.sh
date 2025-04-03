#!/bin/bash
# unified_job_submit.sh

source vsc_utils.sh

# Default configuration
CLUSTER="wice"  # Default cluster
PARTITION="gpu_a100"
TIME="24:00:00"
ACCOUNT=""  # Must be set by user
CPUS_PER_TASK=4
GPUS_PER_NODE=1

function show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -c, --cluster CLUSTER     Cluster (wice/genius)"
    echo "  -p, --partition PART      Slurm partition"
    echo "  -t, --time TIME           Job time limit"
    echo "  -a, --account ACCOUNT     VSC project account"
    echo "  -j, --job-script SCRIPT   Job submission script"
    echo "  -h, --help                Show this help"
    exit 1
}

# Parse arguments (similar to submit_jobs.sh)
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--cluster) CLUSTER="$2"; shift 2 ;;
        -p|--partition) PARTITION="$2"; shift 2 ;;
        -t|--time) TIME="$2"; shift 2 ;;
        -a|--account) ACCOUNT="$2"; shift 2 ;;
        -j|--job-script) JOB_SCRIPT="$2"; shift 2 ;;
        -h|--help) show_help ;;
        *) echo "Unknown option: $1"; show_help ;;
    esac
done

# Validate required parameters
if [[ -z "$ACCOUNT" || -z "$JOB_SCRIPT" ]]; then
    log_error "Account and job script are required"
    show_help
fi

# Create Slurm submission script
sbatch \
    --clusters="$CLUSTER" \
    --partition="$PARTITION" \
    --time="$TIME" \
    --account="$ACCOUNT" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --gpus-per-node="$GPUS_PER_NODE" \
    "$JOB_SCRIPT"