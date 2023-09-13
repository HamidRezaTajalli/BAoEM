import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# local imports
import models
from dataset_handler.data_utils import bootstrap_sample, get_datasets_simple
from training_utils import train_one_epoch, evaluate_one_epoch, test_ensemble_bagging


def bagging_ensemble(ds_name: str, n_models: int = 5, n_epochs: int = 5, model_name: str = 'resnet18'):

    train_batch_size, validation_batch_size = 128, 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = models.get_num_classes(ds_name)
    in_channels = models.get_input_channels(ds_name)

    models_list = [models.get_model(model_name, pretrained=True, num_classes=num_classes, in_channels=in_channels).to(device) for _ in
                   range(n_models)]
    optimizers = [optim.Adam(model.parameters(), weight_decay=1e-4) for model in models_list]
    criterion = nn.CrossEntropyLoss()
    train_dataset, validation_dataset, test_dataset, classes_names = get_datasets_simple()
    validation_dataloader = DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=validation_batch_size, shuffle=True)

    # Define Bagging Strategy
    for i, model in enumerate(models_list):
        print(f"Model {i + 1}\n-------------------------------")
        # Resample the data (with replacement)
        resampled_dataset = bootstrap_sample(train_dataset)
        train_dataloader = DataLoader(resampled_dataset, batch_size=train_batch_size, shuffle=True)

        # Train the model on the resampled data
        for epoch in range(n_epochs):
            print(f"Epoch {epoch + 1}\n-------------------------------")
            train_lss = train_one_epoch(model, train_dataloader, optimizers[i], criterion, device)
            print_string = f"train loss: {train_lss:>6}"
            print(print_string)

            # Test the model on the validation data
            val_lss, val_acc = evaluate_one_epoch(model, validation_dataloader, criterion, device)
            print_string = f"validation loss: {val_lss:>6}"
            print(print_string)
            print_string = f"validation accuracy: {val_acc:>6}"
            print(print_string)

    # Test the ensemble on the test data
    test_acc = test_ensemble_bagging(models_list, device, test_dataloader, criterion)
    print("------------Ensemble Test Accuracy---------")
    print_string = f"test accuracy: {test_acc:>6}"
    print(print_string)
    print("-------------------------------------------")
