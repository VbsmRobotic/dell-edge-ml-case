#!/usr/bin/env bash

# Enable virtual environment usage
USE_VIRTUAL_ENV="TRUE"

if [ "$USE_VIRTUAL_ENV" = "TRUE" ]; then

    # Define the environment directory
    ENV_DIR="env_dell_caseStudy"

    # Check if virtual environment already exists
    if [ ! -f "$ENV_DIR/bin/activate" ]; then

        echo "[INFO] Creating virtual environment..."

        # Install virtualenv if not already available
        python3 -m pip install --user virtualenv

        # Create virtual environment with access to system site packages
        python3 -m virtualenv --system-site-packages "$ENV_DIR"

        # Write activation command to a file for convenience
        echo "source $(pwd)/$ENV_DIR/bin/activate" > source_env_dell_caseStudy.bash

        # Activate virtual environment for current shell
        source "$ENV_DIR/bin/activate"

        # Upgrade pip to a version compatible with Python 3.6.9
        pip install --upgrade pip==20.3.4

        # Install dependencies
        pip install -r requirements.txt

        echo "[INFO] Virtual environment setup complete."
    else
        echo "[INFO] Virtual environment already exists. You can activate it with:"
        echo "source $ENV_DIR/bin/activate"
    fi

fi

