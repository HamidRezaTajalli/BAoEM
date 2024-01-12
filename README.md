This repository contains various ensemble techniques for training models. To execute regular training, run the ensemble technique with the corresponding file. Please note that some files are still under development and may not accept arguments. However, you are free to modify the Python files as per your requirements.

To execute the backdoor version of the ensemble technique, run the file with the technique name followed by the 'backdoor' phrase. Most of these files accept arguments, but you can modify them as needed. Currently, the only available attack is the simple BadNet. For CIFAR-10, the trigger size is 2, and for GTSRB, it is 8.

If you wish to backdoor a single model, you can use the `bdtraining_single_model.py` file. Please note that this file is still under development, so modifications may be necessary based on your specific training requirements.

To replace a backdoored model in a stacking ensemble and retrain, use either `backdooring_while_train_res` or `backdooring_while_train` for VGG19. These files are also under development, so please ensure to modify them before execution.

