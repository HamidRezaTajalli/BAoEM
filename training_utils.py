import torch
from tqdm import tqdm


def train_one_epoch(model, dataloader, optimizer, loss_function, device):
    model = model.to(device)
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        total_samples += data.size(0)

    avg_loss = running_loss / total_samples
    return avg_loss


def evaluate_one_epoch(model, dataloader, loss_function, device):
    model = model.to(device)
    model.eval()
    running_loss = 0.0
    total_samples = 0
    correct_predictions = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_function(output, target)

            running_loss += loss.item() * data.size(0)
            total_samples += data.size(0)

            _, predicted = torch.max(output.data, 1)
            correct_predictions += (predicted == target).sum().item()

    avg_loss = running_loss / total_samples
    accuracy = (correct_predictions / total_samples) * 100
    return avg_loss, accuracy


def stack_outputs(models, batch, device):
    """
    concats the probablity output of multiple models to 1 vector which is later used as an input for meta model. for a given batch of data, the output would have 
    the shape of (batch_size, num_classes * num_models).
    Each model's output is computed with softmax function to convert it into a probability distribution, 
    and then all the outputs are concatenated along the second dimension (dim=1).

    Args:
        models (list): A list of PyTorch models.
        batch (Tensor): A batch of input data.

    Returns:
        Tensor: The stacked output of all models.
    """
    outputs = []
    for model in models:
        model = model.to(device)
        batch = batch.to(device)
        model.eval()
        with torch.no_grad():
            output = torch.softmax(model(batch), dim=1)
            outputs.append(output)
    stacked_output = torch.cat(outputs, dim=1)
    return stacked_output

def train_stacked_models(dataloader, models, n_epochs, optimizers, loss_function, device, save_plot=False, plot_path='loss_plot.png'):
    import matplotlib.pyplot as plt
    losses = {i: [] for i in range(len(models))}  # Initialize a dictionary to store losses for each model
    for epoch in range(n_epochs):
        for i, (model, optimizer) in enumerate(zip(models, optimizers)):
            avg_loss = train_one_epoch(model, dataloader, optimizer, loss_function, device)
            losses[i].append(avg_loss)  # Append the average loss of the current epoch to the corresponding model
    
    # Plotting the losses
    for i, loss in losses.items():
        plt.plot(loss, label=f'Model {i+1}')
    plt.title('Average Loss per Epoch for Each Model')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.legend()
    # plt.show()
    
    # Save the plot if save_plot is True
    if save_plot:
        plt.savefig(plot_path)



def test_ensemble_bagging(models_list, device, test_loader):
    for model in models_list:
        model.eval()

    total_corrects = 0
    total_samples = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_loader, desc="Testing")):
            data, target = data.to(device), target.to(device)
            outputs = torch.zeros(len(target),
                                  len(models_list))  # A matrix to store each model's prediction for each input

            # Get each model's prediction for each input batch
            for model_idx, model in enumerate(models_list):
                output = model(data)
                outputs[:, model_idx] = torch.max(output, dim=1)[1]

            # Use a majority vote system to get final predictions
            final_predictions, _ = torch.mode(outputs, dim=1)
            final_predictions = final_predictions.to(device)
            corrects = torch.sum(final_predictions == target).item()
            total_corrects += corrects
            total_samples += target.size(0)

    accuracy = total_corrects / total_samples
    return accuracy