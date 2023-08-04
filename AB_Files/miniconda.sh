#! /bin/bash
#MIT License
#
#Copyright (c) 2023 Abraham J. Basurto Becerra
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

# --- GLOBAL VARIABLES ---
CONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# CONDA_ENV and BASE_DIR should be provided by the parent script

# --- MINICONDA ENVIRONMENT ---
# Create base directory
if [ ! -d "$BASE_DIR" ]; then
  if ! mkdir -p "$BASE_DIR"; then
    echo "ERROR: Could not create base directory"
    exit 1
  fi
fi
# Install Miniconda
conda_dir="${BASE_DIR}/miniconda3"
if [ ! -d "$conda_dir" ]; then
  conda_installer="${BASE_DIR}/miniconda3.sh"
  if ! wget -q -O "$conda_installer" "$CONDA_URL"; then
    echo "ERROR: Could not download miniconda installer"
    exit 1
  fi
  bash "$conda_installer" -b -p "${conda_dir}/" -s
  rm "$conda_installer"
fi
# Create Conda environment
if [ ! -d "${conda_dir}/envs/${CONDA_ENV}" ]; then
  env_src="${HOME}/conda/${CONDA_ENV}.yml"
  if [ ! -f "$env_src" ]; then
    echo "ERROR: Could not find conda environment file"
    exit 1
  fi
  source "${conda_dir}/bin/activate"
  conda update -q -y conda
  conda env create -q -f "$env_src"
  conda deactivate
fi
# Activate Conda Environment
source "${conda_dir}/bin/activate"
conda activate "$CONDA_ENV"
