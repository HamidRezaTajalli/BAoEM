from stacking import stack_ensemble
from voting import voting_ensemble
import torch

def main():
    # Define the parameters for stack_ensemble
    k_fold = None
    dataname = 'gtsrb'  # replace with your dataset name
    batch_size = 128
    n_epochs = 20
    models_name_list = ['resnet50', 'vgg19', 'efficientnet-b3']  # replace with your model names
    is_pretrained_list = [True, True, True]  # replace with your pretrained flags
    optim_list = ['adam', 'adam', 'adam'] # replace with your optimizers
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mt_mdl_name = 'advanced_4000'
    tr_vl_split = 0.8

    # # Call the stack_ensemble function
    # stack_ensemble(k_fold, dataname, batch_size, n_epochs, models_name_list, 
    #                is_pretrained_list, optim_list, device, mt_mdl_name, tr_vl_split)

    # Call the voting_ensemble function
    voting_ensemble(dataname=dataname, batch_size=batch_size, n_epochs=n_epochs, 
                    models_name_list=models_name_list, is_pretrained_list=is_pretrained_list, 
                    optim_list=optim_list, device=device, tr_vl_split=tr_vl_split, strategy='hard')

if __name__ == "__main__":
    main()




# TODO: save results in csv file. save the models. add load option.
