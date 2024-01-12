import argparse
import torch
from typing import List
from pathlib import Path

from dataset_handler.data_utils import get_datasets, poison_dataset, get_general_transform, get_post_poison_transform, get_pre_poison_transform
from training_utils import train_one_epoch, evaluate_one_epoch, vote
from models import get_num_classes, get_input_channels, get_model
from voting import voting_ensemble
from bagging import bagging_ensemble
import torch.optim as optim

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
 
class MaskedLayer(nn.Module):
    def __init__(self, base, mask):
        super(MaskedLayer, self).__init__()
        self.base = base
        self.mask = mask

    def forward(self, input):
        return self.base(input) * self.mask


parser = argparse.ArgumentParser()
parser.add_argument('--dataname', type=str, default='cifar10')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n_epochs', type=int, default=20)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--device', type=str)
parser.add_argument('--type', type=str, default='voting')
parser.add_argument('--trigger_size', type=int, default=2)
parser.add_argument('--strategy', type=str, default='hard')
parser.add_argument('--ensemble_size', type=int, default=3)
parser.add_argument('--pretrained', type=bool, default=True)
parser.add_argument('--saving_path', type=str, default='.')
parser.add_argument('--experim_num', type=int, default=0)
parser.add_argument('--model_path', type=str, default='.')
parser.add_argument('--pruning_rate', type=float, default=0.5)

args = parser.parse_args()

def main():
    """
        The fine-pruning defense deactivates a fraction of the neurons in the
        network (from the last convolutional layer). The fraction is determined by the pruning rate. The neurons
        are selected based on their activity during training. The neurons with
        the highest activity are deactivated. For evaluating that, we do a forward pass
        through the network and save the activations of the neurons. Then we
        deactivate the neurons with the highest activations.
    """
    
    base_model_list = ['resnet50', 'vgg19', 'efficientnet-b3']
    models_name_list = base_model_list * (args.ensemble_size // len(base_model_list))
    if args.ensemble_size % len(base_model_list) != 0:
        models_name_list += base_model_list[:args.ensemble_size % len(base_model_list)]
    optim_list = []    
    if args.optim.lower() == 'adam':
        optim_list = ['adam'] * args.ensemble_size
    elif args.optim.lower() == 'sgd':
        optim_list = ['sgd'] * args.ensemble_size
    else:
        raise Exception("Invalid optimizer name. Please choose either 'adam' or 'sgd'.")
    
    if args.device:
        if args.device not in ['cpu', 'cuda']:
            raise Exception("Invalid device name. Please choose either 'cpu' or 'cuda'.")
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    is_pretrained_list = [args.pretrained] * args.ensemble_size

    saving_path = Path(args.saving_path)


    num_classes = get_num_classes(args.dataname)
    n_in_channels = get_input_channels(args.dataname)

    # Create a list of models
    model_list = [get_model(model_name, num_classes, n_in_channels, pretrained) for model_name, pretrained in zip(models_name_list, is_pretrained_list)]

    # Define the optimizers
    optim_dict = {'sgd': optim.SGD, 'adam': optim.Adam}
    optimizers = [optim_dict[optim_name](model.parameters(), lr=args.lr) for model, optim_name in zip(model_list, optim_list)]

    criterion = CrossEntropyLoss()

    train_dataset, test_dataset, classes_names = get_datasets(dataname=args.dataname, root_path=None, tr_vl_split=None, transform=get_pre_poison_transform(args.dataname))
    poisoned_trainset = poison_dataset(args, dataname=args.dataname, dataset=train_dataset, is_train=True, attack_name='badnet', post_transform=get_post_poison_transform(args.dataname))
    poisoned_testset = poison_dataset(args, dataname=args.dataname, dataset=test_dataset, is_train=False, attack_name='badnet', post_transform=get_post_poison_transform(args.dataname))
    train_dataset.set_transform(get_general_transform(args.dataname))
    test_dataset.set_transform(get_general_transform(args.dataname))

    # train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=2)
    poisoned_train_dataloader = DataLoader(poisoned_trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    poisoned_test_dataloader = DataLoader(poisoned_testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Evaluate the model on clean and poisoned test data
    print("======== Evaluating the models on clean and poisoned test data ========")
    test_acc = vote(model_list, device, test_dataloader, voting=args.strategy, num_classes=num_classes)
    print(f"test accuracy before re-training: {test_acc:>6}")
    poisoned_test_acc = vote(model_list, device, poisoned_test_dataloader, voting=args.strategy, num_classes=num_classes)
    print(f"poisoned test accuracy before re-training: {poisoned_test_acc:>6}")

    # Get the last convolutional layer of the model, as explained in the paper
    print("======== pruning... ========")
    last_conv_layer_list = []
    for model in model_list:
        layer_to_prune = None
        if model == 'resnet50':
            last_conv_layer_list.append(model.layer4[-1])
            layer_to_prune = 'layer4'
        elif model == 'vgg19':
            last_conv_layer_list.append(model.features[-1])
            layer_to_prune = 'features'
        elif model == 'efficientnet-b3':
            last_conv_layer_list.append(model._conv_head)
            layer_to_prune = '_conv_head'
        else:
            raise Exception("Invalid model name. Please choose one of the following: 'resnet50', 'vgg19', 'efficientnet-b3'.")
        
        with torch.no_grad():
            container = []

            def forward_hook(module, input, output):
                container.append(output)

            hook = getattr(model, layer_to_prune).register_forward_hook(
                forward_hook)
            
            model.eval()
            for data, _ in test_dataloader:
                model(data.cuda())
            hook.remove()

        container = torch.cat(container, dim=0)
        activation = torch.mean(container, dim=[0, 2, 3])
        seq_sort = torch.argsort(activation)
        num_channels = len(activation)
        prunned_channels = int(num_channels * args.pruning_rate)
        mask = torch.ones(num_channels).cuda()
        for element in seq_sort[:prunned_channels]:
            mask[element] = 0
        if len(container.shape) == 4:
            mask = mask.reshape(1, -1, 1, 1)
        setattr(model, layer_to_prune, MaskedLayer(
            getattr(model, layer_to_prune), mask))
        
    
    # Evaluate the models after pruning but before re-training
    
    print("======== Evaluating the models after pruning but before re-training ========")
    test_acc = vote(model_list, device, test_dataloader, voting=args.strategy, num_classes=num_classes)
    print(f"test accuracy before re-training: {test_acc:>6}")
    poisoned_test_acc = vote(model_list, device, poisoned_test_dataloader, voting=args.strategy, num_classes=num_classes)
    print(f"poisoned test accuracy before re-training: {poisoned_test_acc:>6}")

    # Re-train the models on clean data for 10% of the total number of epochs
    print("======== Re-training the models on clean data ========")
    if args.type == 'voting':
        test_acc = voting_ensemble(args.dataname, args.batch_size, args.n_epochs // 10, models_name_list, is_pretrained_list, optim_list, device, tr_vl_split=0.8, strategy=args.strategy)
    elif args.type == 'bagging':
        test_acc = bagging_ensemble(args.dataname, args.batch_size, args.n_epochs // 10, models_name_list, is_pretrained_list, optim_list, device, tr_vl_split=0.8)
    else:
        raise Exception("Invalid ensemble type. Please choose either 'voting' or 'bagging'.")
    
    # Evaluate the models after re-training
    print("======== Evaluating the models after re-training ========")
    test_acc = vote(model_list, device, test_dataloader, voting=args.strategy, num_classes=num_classes)
    print(f"test accuracy after re-training: {test_acc:>6}")
    poisoned_test_acc = vote(model_list, device, poisoned_test_dataloader, voting=args.strategy, num_classes=num_classes)
    print(f"poisoned test accuracy after re-training: {poisoned_test_acc:>6}")

        
if __name__ == 'main':
    main()