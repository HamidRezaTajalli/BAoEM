from stacking import stack_ensemble
import torch

def main():
    # Define the parameters for stack_ensemble
    k_fold = 5
    dataname = 'cifar10'  # replace with your dataset name
    batch_size = 128
    n_epochs = 15
    models_name_list = ['resnet18', 'vgg19', 'resnet18']  # replace with your model names
    is_pretrained_list = [True, False]  # replace with your pretrained flags
    optim_list = ['sgd', 'adam', 'sgd'] # replace with your optimizers
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mt_mdl_name = 'simple_2000'
    tr_vl_split = 0.8

    # Call the stack_ensemble function
    stack_ensemble(k_fold, dataname, batch_size, n_epochs, models_name_list, 
                   is_pretrained_list, optim_list, device, mt_mdl_name, tr_vl_split)

if __name__ == "__main__":
    main()