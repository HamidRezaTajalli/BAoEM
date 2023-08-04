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

#SBATCH --job-name=ASCAD_VAR_ATTACK
#SBATCH --account=icis
#SBATCH --partition=icis
#SBATCH --qos=icis-large                         # see https://wiki.icis-intra.cs.ru.nl/Cluster#Job_Class_Specifications
#SBATCH --nodes=1                                # node count
#SBATCH --nodelist=cn114                         # run in this specific node
#SBATCH --array=1-12
#SBATCH --cpus-per-task=1                        # cpu-cores per task
#SBATCH --mem-per-cpu=2G                         # memory per cpu-core
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/%u/log/slurm/%J.out       # stdout output file
#SBATCH --error=/home/%u/log/slurm/%J.err        # stderr output file
#SBATCH --mail-type=END,FAIL                     # send email when job ends or fails
#SBATCH --mail-user=abasurto@cs.ru.nl

# --- GLOBAL VARIABLES ---
BASE_DIR=/scratch/${USER}
CONDA_ENV=tf-cuda # Conda environment file path: /home/user/conda/env_name.yml


umask 027 # Make sure folders and files are created with sensible permissions
unset XDG_RUNTIME_DIR

# Activate Conda environment
source miniconda.sh

# --- COMMANDS TO BE EXECUTED ---
# Include TensorFlow libraries
CUDNN_PATH="$(dirname "$(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)")")"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib
# Add custom modules to sys.path
export PYTHONPATH="${HOME}/src/:$PYTHONPATH"
# Change to code directory
cd "${HOME}/src/SCA_FLR" || exit 1
# Calculate GE
echo "TASK_ID = $SLURM_ARRAY_TASK_ID"
case $SLURM_ARRAY_TASK_ID in
    1)
      # Finished
        python perform_attack.py ASCAD_variable normal mlp cer ID --attack
            ;;
    2)
        python perform_attack.py ASCAD_variable normal mlp cce ID --attack
    ;;
    3)
      # Finished
        python perform_attack.py ASCAD_variable desync50 mlp cer ID --attack
            ;;
    4)
        python perform_attack.py ASCAD_variable desync50 mlp cce ID --attack
    ;;
    5)
      # Finished
        python perform_attack.py ASCAD_variable desync100 mlp cer ID --attack
            ;;
    6)
        python perform_attack.py ASCAD_variable desync100 mlp cce ID --attack
    ;;
    7)
      # Finished
        python perform_attack.py ASCAD_variable normal cnn cer ID --attack
            ;;
    8)
        python perform_attack.py ASCAD_variable normal cnn cce ID --attack
    ;;
    9)
      # Finished
        python perform_attack.py ASCAD_variable desync50 cnn cer ID --attack
            ;;
    10)
        python perform_attack.py ASCAD_variable desync50 cnn cce ID --attack
    ;;
    11)
      # Finished
        python perform_attack.py ASCAD_variable desync100 cnn cer ID --attack
            ;;
    12)
        python perform_attack.py ASCAD_variable desync100 cnn cce ID --attack
    ;;
    13)
      # Error
        python perform_attack.py ASCAD_variable normal mlp cer HW --attack
            ;;
    14)
        python perform_attack.py ASCAD_variable normal mlp cce HW --attack
    ;;
    15)
        python perform_attack.py ASCAD_variable desync50 mlp cer HW --attack
            ;;
    16)
        python perform_attack.py ASCAD_variable desync50 mlp cce HW --attack
    ;;
    17)
        python perform_attack.py ASCAD_variable desync100 mlp cer HW --attack
            ;;
    18)
        python perform_attack.py ASCAD_variable desync100 mlp cce HW --attack
    ;;
    19)
      # Error
        python perform_attack.py ASCAD_variable normal cnn cer HW --attack
            ;;
    20)
        python perform_attack.py ASCAD_variable normal cnn cce HW --attack
    ;;
    21)
        python perform_attack.py ASCAD_variable desync50 cnn cer HW --attack
            ;;
    22)
        python perform_attack.py ASCAD_variable desync50 cnn cce HW --attack
    ;;
    23)
        python perform_attack.py ASCAD_variable desync100 cnn cer HW --attack
            ;;
    24)
        python perform_attack.py ASCAD_variable desync100 cnn cce HW --attack
    ;;
    *)
        echo -n "Unknown TASK_ID"
    ;;
esac

