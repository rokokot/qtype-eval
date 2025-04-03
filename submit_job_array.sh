# Near the beginning of the file, update:
CLUSTER="genius"  # or "wice" depending on which you're using

# When creating the script, add proper module loading:
cat >> $EXPERIMENT_SCRIPT << EOF
#!/bin/bash
#SBATCH --job-name=qtype-exp
...

# Load modules
module purge
module load Python/3.9

# Set environment variables
export PYTHONPATH=\$PYTHONPATH:$PWD
export HF_HOME=\$VSC_SCRATCH/hf_cache
EOF