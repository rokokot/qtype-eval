function setup_vsc_environment() {
    # Ensure we're in the project root
    PROJECT_ROOT="$(pwd)"
    
    # Common module and path setup
    module purge
    module load Python/3.9

    # Set up Miniconda if not exists
    if [ ! -d "$VSC_DATA/miniconda3" ]; then
        echo "Setting up Miniconda..."
        cd $VSC_DATA
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
        bash Miniconda3-latest-Linux-x86_64.sh -b -p $VSC_DATA/miniconda3
        rm Miniconda3-latest-Linux-x86_64.sh
        cd "$PROJECT_ROOT"  # Return to project root
    fi

    # Common environment variables
    export PATH="$VSC_DATA/miniconda3/bin:$PATH"
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
    export HF_HOME=$VSC_SCRATCH/hf_cache

    # Create necessary directories
    mkdir -p data/cache data/features outputs logs
}