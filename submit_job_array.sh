#!/bin/bash

ACCOUNT=""
PARTITION="gpu"
TIME="24:00:00"
MAX_JOBS=100
START_ID=1
END_ID=100
CLUSTER="genius"    
MEMORY_PER_CPU=8  
CPUS_PER_TASK=4
GPUS_PER_NODE=1
MAIL_TYPE="NONE"  # : NONE, BEGIN, END, FAIL, ALL
MAIL_USER=""

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' 

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
        --cpus)
            CPUS_PER_TASK="$2"
            shift 2
            ;;
        --mem-per-cpu)
            MEMORY_PER_CPU="$2"
            shift 2
            ;;
        --gpus)
            GPUS_PER_NODE="$2"
            shift 2
            ;;
        --mail-type)
            MAIL_TYPE="$2"
            shift 2
            ;;
        --mail-user)
            MAIL_USER="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --account ACCOUNT   Slurm account (required)"
            echo "  --partition PART    Slurm partition (default: gpu)"
            echo "  --time TIME         Walltime limit (default: 24:00:00)"
            echo "  --max-jobs N        Maximum concurrent jobs (default: 100)"
            echo "  --start N           Starting experiment ID (default: 1)"
            echo "  --end N             Ending experiment ID (default: 100)"
            echo "  --cluster NAME      Cluster name (default: wice)"
            echo "  --cpus N            CPUs per task (default: 4)"
            echo "  --mem-per-cpu N     Memory per CPU in GB (default: 8)"
            echo "  --gpus N            GPUs per node (default: 1)"
            echo "  --mail-type TYPE    Mail notification type (NONE, BEGIN, END, FAIL, ALL)"
            echo "  --mail-user EMAIL   Email address for notifications"
            echo "  -h, --help          Show this help"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

if [ -z "$ACCOUNT" ]; then
    echo -e "${RED}Error: --account is required${NC}"
    exit 1
fi

mkdir -p logs
mkdir -p checkpoints
mkdir -p outputs

TOTAL_EXPERIMENTS=$((END_ID - START_ID + 1))
echo -e "${GREEN}Preparing to submit $TOTAL_EXPERIMENTS experiments${NC}"

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
#SBATCH --cpus-per-task=$CPUS_PER_TASK
#SBATCH --mem-per-cpu=${MEMORY_PER_CPU}G
#SBATCH --gpus-per-node=$GPUS_PER_NODE
EOF

if [ "$MAIL_TYPE" != "NONE" ]; then
    if [ -z "$MAIL_USER" ]; then
        echo -e "${YELLOW}Warning: mail-type specified but no mail-user. Notifications will not be sent.${NC}"
    else
        cat >> $EXPERIMENT_SCRIPT << EOF
#SBATCH --mail-type=$MAIL_TYPE
#SBATCH --mail-user=$MAIL_USER
EOF
    fi
fi

cat >> $EXPERIMENT_SCRIPT << EOF

# Create necessary directories
mkdir -p logs
mkdir -p checkpoints
mkdir -p outputs

# Set environment variables
export OMP_NUM_THREADS=$CPUS_PER_TASK

# Create a unique identifier for this experiment
EXPERIMENT_ID=\$SLURM_ARRAY_TASK_ID
CHECKPOINT_FILE="checkpoints/exp_\${EXPERIMENT_ID}.done"
TIMESTAMP=\$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/exp_\${SLURM_ARRAY_JOB_ID}_\${EXPERIMENT_ID}_\${TIMESTAMP}.log"

# Redirect all output to the log file
exec > >(tee -a "\$LOG_FILE") 2>&1

echo "======================================================"
echo "Job ID: \$SLURM_JOB_ID, Array Task ID: \$EXPERIMENT_ID"
echo "Running on: \$(hostname)"
echo "Start time: \$(date)"
echo "======================================================"

# Check if this experiment was already completed
if [ -f "\$CHECKPOINT_FILE" ]; then
    echo "Experiment \$EXPERIMENT_ID already completed. Skipping."
    exit 0
fi

# Try to get the experiment configuration
echo "Getting configuration for experiment ID \$EXPERIMENT_ID..."
CONFIG=\$(python experiments/get_experiment_config.py \$EXPERIMENT_ID)

if [ \$? -ne 0 ]; then
    echo "Error getting configuration for experiment ID \$EXPERIMENT_ID"
    exit 1
fi

echo "Running experiment with config: \$CONFIG"

# Set the experiment-specific output directory
OUTPUT_DIR="outputs/exp_\${EXPERIMENT_ID}_\${TIMESTAMP}"
mkdir -p "\$OUTPUT_DIR"

# Run the experiment
echo "Starting experiment at \$(date)"
SECONDS=0
python -m src.experiments.run_experiment \$CONFIG output_dir="\$OUTPUT_DIR"
RESULT=\$?
DURATION=\$SECONDS

# Check the result
if [ \$RESULT -eq 0 ]; then
    echo "Experiment \$EXPERIMENT_ID completed successfully in \$(printf '%02d:%02d:%02d' \$((\$DURATION/3600)) \$((\$DURATION%3600/60)) \$((\$DURATION%60)))."
    # Create checkpoint file to mark completion
    touch "\$CHECKPOINT_FILE"
    echo "\$(date)" > "\$CHECKPOINT_FILE"
    exit 0
else
    echo "Experiment \$EXPERIMENT_ID failed with error code \$RESULT after \$(printf '%02d:%02d:%02d' \$((\$DURATION/3600)) \$((\$DURATION%3600/60)) \$((\$DURATION%60)))."
    exit 1
fi
EOF

chmod +x $EXPERIMENT_SCRIPT

echo -e "${GREEN}Job array configuration:${NC}"
echo "  Account:       $ACCOUNT"
echo "  Partition:     $PARTITION"
echo "  Cluster:       $CLUSTER"
echo "  Time limit:    $TIME"
echo "  CPUs per task: $CPUS_PER_TASK"
echo "  Memory/CPU:    ${MEMORY_PER_CPU}G"
echo "  GPUs/node:     $GPUS_PER_NODE"
echo "  Array range:   $START_ID-$END_ID"
echo "  Max jobs:      $MAX_JOBS"

echo -e "${GREEN}Submitting job array for experiments $START_ID to $END_ID${NC}"
JOBID=$(sbatch --array=$START_ID-$END_ID%$MAX_JOBS $EXPERIMENT_SCRIPT | awk '{print $4}')

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Job array submitted with ID: $JOBID${NC}"
    echo "Use 'squeue -j $JOBID' to check status"
    echo "Use 'scancel $JOBID' to cancel the entire job array"
    echo "Logs will be written to logs/exp_${JOBID}_*.log"
    
    SUMMARY_FILE="submissions_${JOBID}.txt"
    cat > $SUMMARY_FILE << SUMMARY
Job Array ID: $JOBID
Submitted at: $(date)
Account: $ACCOUNT
Partition: $PARTITION
Cluster: $CLUSTER
Array range: $START_ID-$END_ID
Max concurrent jobs: $MAX_JOBS
SUMMARY
    
    echo "Summary saved to $SUMMARY_FILE"
else
    echo -e "${RED}Error submitting job array${NC}"
    exit 1
fi