from copy import deepcopy
from typing import List
import torch, torchvision
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
from torchvision import transforms

from dataset_handler.data_utils import get_datasets, poison_dataset, get_general_transform, get_post_poison_transform, get_pre_poison_transform, create_stacked_dataset
from training_utils import train_stacked_models, train_one_epoch, stack_outputs
from models import get_num_classes, get_input_channels, get_model

import torch.optim as optim


def backdoor() -> None:
    
    # Get the number of classes and input channels from the dataset
    dataname = 'gtsrb'
    batch_size = 64
    n_epochs = 20
    epsilon = 0.02
    trigger_scale = 0.08
    color = 'green'
    target_label = 0
    num_classes = get_num_classes(dataname)
    n_in_channels = get_input_channels(dataname)
    lr = 0.001   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ensemble_size = 50


    base_model_list = ['resnet50', 'vgg19', 'efficientnet-b3']
    models_name_list = base_model_list * (ensemble_size // len(base_model_list))
    if ensemble_size % len(base_model_list) != 0:
        models_name_list += base_model_list[:ensemble_size % len(base_model_list)]


    k_fold = None
    is_pretrained_list = [True] * ensemble_size  # replace with your pretrained flags
    optim_list = ['adam'] * ensemble_size # replace with your optimizers
    mt_mdl_name = 'advanced_4000'
    tr_vl_split = 0.8

#######################################################################trial########################################
    # model = get_model('resnet50', num_classes, n_in_channels, True)

    # model = model.to(device)
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    # loss_function = CrossEntropyLoss()
    

    # trainset, testset, classes_names = get_datasets(dataname=dataname, root_path=None, tr_vl_split=None, transform=transforms.ToTensor())
    # poisoned_trainset = poison_dataset(dataname=dataname, dataset=trainset, is_train=True, attack_name='badnet', post_transform=get_post_poison_transform_cifar10())
    # poisoned_testset = poison_dataset(dataname=dataname, dataset=testset, is_train=False, attack_name='badnet', post_transform=get_post_poison_transform_cifar10())
    # testset.set_transform(get_general_transform_cifar10())
    # trainset.set_transform(get_general_transform_cifar10())

    # import matplotlib.pyplot as plt
    # import numpy as np

    # def save_image(img, labels, filename):
    #     img = img / 2 + 0.5     # unnormalize
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.title(' '.join('%5s' % classes_names[labels[j]] for j in range(5)))
    #     plt.savefig(filename)

    # # get some random training images
    # dataiter = iter(DataLoader(trainset, batch_size=5, shuffle=True))
    # images, labels = next(dataiter)
    # save_image(torchvision.utils.make_grid(images), labels, 'trainset.png')

    # dataiter = iter(DataLoader(testset, batch_size=5, shuffle=True))
    # images, labels = next(dataiter)
    # save_image(torchvision.utils.make_grid(images), labels, 'testset.png')

    # dataiter = iter(DataLoader(poisoned_trainset, batch_size=5, shuffle=True))
    # images, labels = next(dataiter)
    # save_image(torchvision.utils.make_grid(images), labels, 'poisoned_trainset.png')

    # dataiter = iter(DataLoader(poisoned_testset, batch_size=5, shuffle=True))
    # images, labels = next(dataiter)
    # save_image(torchvision.utils.make_grid(images), labels, 'poisoned_testset.png')

    # exit()


    # train_dataloader = DataLoader(poisoned_trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    # bd_test_dataloader = DataLoader(poisoned_testset, batch_size=batch_size, shuffle=False, num_workers=2)
    # test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # model.train()
    # for epoch in range(n_epochs):
    #     running_loss = 0.0
    #     correct = 0
    #     total = 0
    #     for i, data in enumerate(train_dataloader, 0):
    #         inputs, labels = data
    #         inputs, labels = inputs.to(device), labels.to(device)

    #         optimizer.zero_grad()

    #         outputs = model(inputs)
    #         loss = loss_function(outputs, labels)
    #         loss.backward()
    #         optimizer.step()

    #         running_loss += loss.item()
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()

    #     print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_dataloader)}, Accuracy: {100 * correct / total}%')
    
    # # Testing the model on bd_test_dataset
    # correct_bd = 0
    # total_bd = 0
    # model.eval()
    # with torch.no_grad():
    #     for data in bd_test_dataloader:
    #         images, labels = data
    #         images, labels = images.to(device), labels.to(device)
    #         outputs = model(images)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total_bd += labels.size(0)
    #         correct_bd += (predicted == labels).sum().item()
    # print(f'Accuracy of the model on the bd_test_dataset: {100 * correct_bd / total_bd}%')

    # # Testing the model on test_dataset
    # correct_test = 0
    # total_test = 0
    # model.eval()
    # with torch.no_grad():
    #     for data in test_dataloader:
    #         images, labels = data
    #         images, labels = images.to(device), labels.to(device)
    #         outputs = model(images)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total_test += labels.size(0)
    #         correct_test += (predicted == labels).sum().item()
    # print(f'Accuracy of the model on the test_dataset: {100 * correct_test / total_test}%')

    # # Save the model
    # from datetime import datetime
    # model_type = type(model).__name__
    # current_date = datetime.now().strftime('%Y-%m-%d')
    # torch.save(model.state_dict(), f'{model_type}_{current_date}.pth')



#######################################################################trial################################################

    model_list = [get_model(model_name, num_classes, n_in_channels, pretrained) for model_name, pretrained in zip(models_name_list, is_pretrained_list)]

    # Define the optimizers
    optim_dict = {'sgd': optim.SGD, 'adam': optim.Adam}
    optimizers = [optim_dict[optim_name](model.parameters(), lr=lr) for model, optim_name in zip(model_list, optim_list)]

    # Define the loss function
    loss_function = CrossEntropyLoss() 

    # Initialize the meta-dataset
    meta_dataset = None

    # If k_fold is None, split the train_dataset into training and validation sets based on tr_vl_split ratio. 
    # then train the base models on the training set and create the meta-dataset using the validation set.
    if k_fold is None:
        # train_dataset, validation_dataset, test_dataset, classes_names = get_datasets(dataname=dataname, tr_vl_split=tr_vl_split)
        train_dataset, validation_dataset, test_dataset, classes_names = get_datasets(dataname=dataname, root_path=None, tr_vl_split=tr_vl_split, transform=get_pre_poison_transform(dataname))
        poisoned_trainset = poison_dataset(dataname=dataname, dataset=train_dataset, is_train=True, attack_name='badnet', post_transform=get_post_poison_transform(dataname))
        poisoned_testset = poison_dataset(dataname=dataname, dataset=test_dataset, is_train=False, attack_name='badnet', post_transform=get_post_poison_transform(dataname))
        poisoned_validationset = poison_dataset(dataname=dataname, dataset=validation_dataset, is_train=True, attack_name='badnet', post_transform=get_post_poison_transform(dataname))
        train_dataset.dataset.set_transform(get_general_transform(dataname))
        test_dataset.set_transform(get_general_transform(dataname))
        validation_dataset.dataset.set_transform(get_general_transform(dataname))

        # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        poisoned_train_dataloader = DataLoader(poisoned_trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        train_stacked_models(poisoned_train_dataloader, model_list, n_epochs, optimizers, loss_function, device)
        # validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        poisoned_validation_dataloader = DataLoader(poisoned_validationset, batch_size=batch_size, shuffle=True, num_workers=2)
        meta_dataset = create_stacked_dataset(model_list, poisoned_validation_dataloader, device)

    # If k_fold is not None, perform k-fold cross-validation and create the meta-dataset using the k-fold cross validation sets.
    # TODO: this section is not modified for badnet. I should modify the datasets correctly.
    else:        
        # train_dataset, test_dataset, classes_names = get_datasets(dataname=dataname, tr_vl_split=None)
        train_dataset, test_dataset, classes_names = get_datasets(dataname=dataname, root_path=None, tr_vl_split=None, transform=get_pre_poison_transform(dataname))
        # poisoned_trainset = poison_dataset(dataname=dataname, dataset=trainset, is_train=True, attack_name='badnet', post_transform=get_post_poison_transform_cifar10())
        poisoned_testset = poison_dataset(dataname=dataname, dataset=test_dataset, is_train=False, attack_name='badnet', post_transform=get_post_poison_transform(dataname))
        train_dataset.set_transform(get_general_transform(dataname))
        test_dataset.set_transform(get_general_transform(dataname))
        meta_dataset_list = []
        kf = KFold(n_splits=k_fold, shuffle=False)
        for train_idx, val_idx in kf.split(train_dataset.data):
            train_subsampler = SubsetRandomSampler(train_idx)
            val_subsampler = SubsetRandomSampler(val_idx)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_subsampler, num_workers=2)
            train_stacked_models(train_dataloader, model_list, n_epochs, optimizers, loss_function, device)
            validation_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_subsampler, num_workers=2)
            meta_dataset_list.append(create_stacked_dataset(model_list, validation_dataloader, device))
        meta_dataset = torch.utils.data.ConcatDataset(meta_dataset_list)

    # Test the performance of models in model_list on validation data
    with torch.no_grad():
        for model in model_list:
            model.eval()
            total_samples = 0
            correct_predictions = 0
            for batch_idx, (data, labels) in enumerate(poisoned_validation_dataloader):
                data, labels = data.to(device), labels.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)
            accuracy = (correct_predictions / total_samples) * 100
            print(f'Validation Accuracy of the model {model.__class__.__name__} on the validation images: {accuracy}%')


    meta_dataloader = DataLoader(meta_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    meta_model = get_model(mt_mdl_name, num_classes, num_classes*len(model_list), False)

    meta_optimizer = optim.Adam(meta_model.parameters(), lr=0.001)

    # Train the meta model
    for epoch in range(n_epochs):
        train_one_epoch(meta_model, meta_dataloader, meta_optimizer, loss_function, device)


    # Test the whole ensemble on the test dataset
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)   
    meta_model.eval()
    total_samples = 0
    correct_predictions = 0
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(test_dataloader):
            data, labels = data.to(device), labels.to(device)
            meta_input = stack_outputs(model_list, data, device)
            output = meta_model(meta_input)
            _, predicted = torch.max(output.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    accuracy = (correct_predictions / total_samples) * 100
    print(f'Ensemble of size: {ensemble_size}. Test Accuracy of the model on the test images: {accuracy}%')

    # Test the whole ensemble on the poisoned test dataset
    test_dataloader = DataLoader(poisoned_testset, batch_size=batch_size, shuffle=True, num_workers=2)   
    meta_model.eval()
    total_samples = 0
    correct_predictions = 0
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(test_dataloader):
            data, labels = data.to(device), labels.to(device)
            meta_input = stack_outputs(model_list, data, device)
            output = meta_model(meta_input)
            _, predicted = torch.max(output.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    accuracy = (correct_predictions / total_samples) * 100
    print(f'Ensemble of size: {ensemble_size}. Test Accuracy of the model on the poisoned test images: {accuracy}%')



if __name__ == "__main__":
    backdoor()



