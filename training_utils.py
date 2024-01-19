import torch
from tqdm import tqdm
import gc


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
        data, target = data.to('cpu'), target.to('cpu')
        torch.cuda.empty_cache()  # Free up memory
    gc.collect()
    model.to('cpu')

    avg_loss = running_loss / total_samples
    return avg_loss

def train_one_epoch_boosting(model, dataloader, optimizer, loss_function, device, prediction, model_index):
    model = model.to(device)
    model.train()
    running_loss = 0.0
    total_samples = 0
    # (n_model,_batch_id, 256, 10)
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # first make a prediction with current model
        output = model(data)
        if len(prediction) != 0:
            #print('add!')
            torch.add(output, prediction[batch_idx])
        loss = loss_function(torch.div(output, model_index+1), target)
        # loss = loss_function(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        total_samples += data.size(0)
        data, target = data.to('cpu'), target.to('cpu')
        torch.cuda.empty_cache()  # Free up memory
    gc.collect()
    model.to('cpu')

    avg_loss = running_loss / total_samples
    return avg_loss

def get_pred_output(model, dataloader, prediction, device):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        if len(prediction) == 0:
            for batch_idx, (data, _) in enumerate(dataloader):
                data = data.to(device)
                output = model(data)
                prediction.append(output)
                output = output.to('cpu')
                data = data.to('cpu')
        else:
            for batch_idx, (data, _) in enumerate(dataloader):
                data = data.to(device)
                output = model(data)
                prediction[batch_idx] += output
                output = output.to('cpu')
                data = data.to('cpu')
    model.to('cpu')
    return prediction

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
            data, target = data.to('cpu'), target.to('cpu')

    avg_loss = running_loss / total_samples
    accuracy = (correct_predictions / total_samples) * 100
    model.to('cpu')
    return avg_loss, accuracy


def stack_outputs(models, batch, device='cpu'):
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
        #if next(model.parameters()).is_cuda:
        #    print("Model is on GPU")
        #    exit()
        #if batch.is_cuda:
        #    print("Data is on GPU")
        #    exit()

        model.eval()
        with torch.no_grad():
            output = torch.softmax(model(batch), dim=1).to('cpu')
            outputs.append(output)
    stacked_output = torch.cat(outputs, dim=1).to('cpu')
    return stacked_output

def train_stacked_models(dataloader, models, n_epochs, optimizers, loss_function, device, save_plot=False, plot_path='loss_plot.png'):
    import matplotlib.pyplot as plt
    losses = {i: [] for i in range(len(models))}  # Initialize a dictionary to store losses for each model
    for epoch in range(n_epochs):
        for i, (model, optimizer) in enumerate(zip(models, optimizers)):
            model.to(device) # Move the model to the device
            print(f"Training model {i+1} for epoch {epoch+1}...")
            avg_loss = train_one_epoch(model, dataloader, optimizer, loss_function, device)
            model.to('cpu') # Move the model back to the CPU
            losses[i].append(avg_loss)  # Append the average loss of the current epoch to the corresponding model
    torch.cuda.empty_cache()  # Free up memory
    gc.collect()
    
    # # Plotting the losses
    # for i, loss in losses.items():
    #     plt.plot(loss, label=f'Model {i+1}')
    # plt.title('Average Loss per Epoch for Each Model')
    # plt.xlabel('Epoch')
    # plt.ylabel('Average Loss')
    # plt.legend()
    # # plt.show()
    
    # # Save the plot if save_plot is True
    # if save_plot:
    #     plt.savefig(plot_path)



def vote(models_list, device, test_loader, voting='hard', num_classes=10):
    for model in models_list:
        model.eval()
        

    total_corrects = 0
    total_samples = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_loader, desc="Testing")):
            data, target = data.to(device), target.to(device)
            if voting == 'hard':
                outputs = torch.zeros(len(target),
                                      len(models_list))  # A matrix to store each model's prediction for each input

                # Get each model's prediction for each input batch
                for model_idx, model in enumerate(models_list):
                    model.to(device)
                    output = model(data)
                    outputs[:, model_idx] = torch.max(output, dim=1)[1]

                # Use a majority vote system to get final predictions
                final_predictions, _ = torch.mode(outputs, dim=1)
            elif voting == 'soft':
                outputs = torch.zeros(len(target),
                                      len(models_list), num_classes)  # A matrix to store each model's prediction for each input

                # Get each model's prediction for each input batch
                for model_idx, model in enumerate(models_list):
                    model.to(device)
                    output = torch.softmax(model(data), dim=1)
                    outputs[:, model_idx, :] = output

                # Use a majority vote system to get final predictions
                final_predictions = torch.argmax(torch.sum(outputs, dim=1), dim=1)
            else:
                raise ValueError("Invalid voting mechanism. Choose either 'hard' or 'soft'.")

            final_predictions = final_predictions.to(device)
            corrects = torch.sum(final_predictions == target).item()
            total_corrects += corrects
            total_samples += target.size(0)

    accuracy = total_corrects / total_samples
    return accuracy
