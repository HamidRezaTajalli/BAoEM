from pathlib import Path
import gc
import argparse
import clean_train


parser = argparse.ArgumentParser(description='BASL_Autoencoder')
parser.add_argument('--dataname', type=str, default='mnist',
                    choices=['mnist', 'svhn', 'fmnist', 'cifar10'],
                    help='The dataset to use')
parser.add_argument('--n_models', type=int, default=5,
                    help='number of ensemble models to train')
parser.add_argument('--n_epochs', type=int, default=100,
                    help='number of epochs to train')
parser.add_argument('--model_name', type=str, default='resnet18',
                    choices=['resnet18'],
                    help="model name to use")

args = parser.parse_args()


saving_path = Path()
n_of_experiments = 1
ds_name = args.dataname
n_models = args.n_models
n_epochs = args.n_epochs
model_name = args.model_name

clean_train.clean_train(ds_name=ds_name, n_models=n_models, n_epochs=n_epochs, model_name=model_name)

