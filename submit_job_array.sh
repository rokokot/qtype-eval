#!/bin/bash
# submit_job_array.sh
ACCOUNT=""
PARTITION="gpu"
TIME="24:00:00"
MAX_JOBS=100
START_ID=1
END_ID=100
CLUSTER="wice"    # genius

while [[ $# -gt 0 ]]; do
    case $1 in
        --account)
            ACCOUNT="$2"
            shift 2
            ;;
        --partition)
            PARTITION="$2"
            shift 2
            ;;
        --time)
            TIME="$2"
            shift 2
            ;;
        --max-jobs)
            MAX_JOBS="$2"
            shift 2
            ;;
        --start)
            START_ID="$2"
            shift 2
            ;;
        --end)
            END_ID="$2"
            shift 2
            ;;
        --cluster)
            CLUSTER="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --account ACCOUNT   Slurm account"
            echo "  --partition PART    Slurm partition (default: gpu)"
            echo "  --time TIME         Walltime limit (default: 24:00:00)"
            echo "  --max-jobs N        Maximum concurrent jobs (default: 100)"
            echo "  --start N           Starting experiment ID (default: 1)"
            echo "  --end N             Ending experiment ID (default: 100)"
            echo "  --cluster NAME      Cluster name (default: wice)"
            echo "  -h, --help          Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$ACCOUNT" ]; then
    echo "Error: --account is required"
    exit 1
fi

EXPERIMENT_SCRIPT="job_array_experiment.sh"
cat > $EXPERIMENT_SCRIPT << EOF
#!/bin/bash
#SBATCH --job-name=qtype-exp
#SBATCH --output=logs/exp_%A_%a.log
#SBATCH --error=logs/exp_%A_%a.log
#SBATCH --time=$TIME
#SBATCH --account=$ACCOUNT
#SBATCH --partition=$PARTITION
#SBATCH --clusters=$CLUSTER
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus-per-node=1

mkdir -p logs

CONFIG=\$(python experiments/get_experiment_config.py \$SLURM_ARRAY_TASK_ID)

if [ \$? -ne 0 ]; then
    echo "Error getting configuration for experiment ID \$SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "Running experiment with config: \$CONFIG"
python -m src.experiments.run_experiment \$CONFIG
EOF

chmod +x $EXPERIMENT_SCRIPT

mkdir -p logs

echo "Submitting job array for experiments $START_ID to $END_ID"
JOBID=$(sbatch --array=$START_ID-$END_ID%$MAX_JOBS $EXPERIMENT_SCRIPT | awk '{print $4}')

if [ $? -eq 0 ]; then
    echo "Job array submitted with ID: $JOBID"
    echo "Use 'squeue -j $JOBID' to check status"
    echo "Logs will be written to logs/exp_${JOBID}_*.log"
else
    echo "Error submitting job array"
    exit 1
fi