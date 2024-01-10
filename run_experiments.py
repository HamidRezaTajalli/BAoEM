import os
from typing import Dict, List

# Define constants
STRATEGY = ['hard', 'soft'] 
ENSEMBLE_SIZE = [10, 25, 50]

TEMPLATE_PATH = "./job_executer.sh"

def replace_template(template: str, strategy: str, ensemble_size: int) -> str:
    template = template.replace('STRATEGY', strategy)
    template = template.replace('ENSEMBLE_SIZE', str(ensemble_size))
    return template

def create_directory(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)

def write_to_file(sbatch_name: str, template: str) -> None:
    with open(sbatch_name, "w") as f:
        f.write(template)

def run_sbatch(sbatch_name: str) -> None:
    os.system("sbatch {}".format(sbatch_name))

def main() -> None:
    for strategy in STRATEGY:
        for ensemble_size in ENSEMBLE_SIZE:
            with open(TEMPLATE_PATH, 'r') as file:
                template = file.read()
            template = replace_template(template, strategy, ensemble_size)
            sbatch_name = f"{strategy}_{ensemble_size}.sh"
            # path = os.path.split(sbatch_name)[0]
            path = 'bash_scripts'
            create_directory(path)
            write_to_file(sbatch_name, template)
            run_sbatch(sbatch_name)

# def main() -> None:
#     strategy = 'hard'
#     for ensemble_size in ENSEMBLE_SIZE:
#             with open(TEMPLATE_PATH, 'r') as file:
#                 template = file.read()
#             template = replace_template(template, strategy, ensemble_size)
#             # sbatch_name = f"{strategy}_{ensemble_size}.sh"
#             sbatch_name = f"bagging_{ensemble_size}.sh"
#             # path = os.path.split(sbatch_name)[0]
#             path = './bash_scripts/'
#             create_directory(path)
#             write_to_file(sbatch_name, template)
#             run_sbatch(sbatch_name)

if __name__ == "__main__":
    main()

