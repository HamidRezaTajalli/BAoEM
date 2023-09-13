import torch
import random
import torch.nn.functional as F
from dataset_handler.cifar10 import get_dataloaders_simple
import kornia.augmentation as A

def get_grids(k, img_size):
    ins = torch.rand(1, 2, k, k) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))
    noise_grid = (
        torch.nn.functional.upsample(ins, size=img_size, mode="bicubic", align_corners=True)
        .permute(0, 2, 3, 1)
    )
    array1d = torch.linspace(-1, 1, steps=img_size)
    x, y = torch.meshgrid(array1d, array1d)
    identity_grid = torch.stack((y, x), 2)[None, ...]
    return noise_grid, identity_grid

def poison_sample(sample, sample_dimension, identity_grid, noise_grid, s: float, grid_rescale: int):
    """
    Applies a transformation to a sample using a noise grid and an identity grid.

    Args:
        sample: The sample to be poisoned.
        sample_dimension (tuple): The dimension of the sample.
        identity_grid: The identity grid used for poisoning.
        noise_grid: The noise grid used for poisoning.
        s (float): The scale factor for the noise grid.
        grid_rescale (int): The rescale factor for the grid.

    Returns:
        The poisoned sample.
    """
    
    grid_temps = (identity_grid + s * noise_grid / sample_dimension[1]) * grid_rescale
    grid_temps = torch.clamp(grid_temps, -1, 1)
    return F.grid_sample(sample.unsqueeze(0), grid_temps, align_corners=True)[0]



def get_poisoned_dataset(is_train: bool, dataset: torch.utils.data.Dataset, sample_dimension: tuple, s: float, k: int,
                         grid_rescale: int, poison_rate: float, cross_ratio: float, noise_grid, identity_grid,
                         target_class: int, num_classes, all_to_all: bool, source_label: int = None):
    """
    Generates a poisoned dataset based on the given parameters.

    Args:
        is_train (bool): If True, returns the poisoned train set, otherwise returns the poisoned test set.
        dataset (torch.utils.data.Dataset): The original dataset to be poisoned.
        sample_dimension (tuple): The dimension of the samples in the dataset.
        s (float): The scale factor for the noise grid.
        k (int): The size of the noise grid.
        grid_rescale (int): The rescale factor for the grid.
        poison_rate (float): The rate at which the dataset is poisoned.
        cross_ratio (float): The ratio of cross noise images in the dataset.
        noise_grid: The noise grid used for poisoning.
        identity_grid: The identity grid used for poisoning.
        target_class (int): The target class for the poisoned samples.
        num_classes: The total number of classes in the dataset.
        all_to_all (bool): If True, all classes are targeted, otherwise only the target class is targeted.
        source_label (int, optional): The source label for the poisoned samples. If None, all labels are considered as source.

    Returns:
        tuple: The poisoned dataset, the labels of the poisoned dataset, the indices of the poisoned samples, and the indices of the cross noise samples.
    """
    n_channels = sample_dimension[0]
    input_width = sample_dimension[1]
    input_height = sample_dimension[2]
    ds_length = len(dataset)

    img_set = []
    label_set = []
    pt = 0
    ct = 0
    cnt = 0

    poison_id = []
    cross_id = []
    poison_indices, cross_indices = None, None

    grid_temps = (identity_grid + s * noise_grid / input_height) * grid_rescale
    grid_temps = torch.clamp(grid_temps, -1, 1)

    # printing for test:
    # print("Shape of grid_temps after clamping:", grid_temps.shape)
    # print("Shape of noise_grid:", noise_grid.shape)
    # print("Shape of identity_grid:", identity_grid.shape)


    # if preparing trainset:
    if is_train:
        # random sampling
        id_set = list(range(0, ds_length))
        random.shuffle(id_set)
        num_poison = int(ds_length * poison_rate)
        poison_indices = id_set[:num_poison]
        poison_indices.sort()  # increasing order

        num_cross = int(ds_length * cross_ratio)
        cross_indices = id_set[num_poison:num_poison + num_cross]  # use **non-overlapping** images to cover
        cross_indices.sort()

        ins = torch.rand(1, input_height, input_height, 2) * 2 - 1
        grid_temps2 = grid_temps + ins / input_height
        grid_temps2 = torch.clamp(grid_temps2, -1, 1)

        #printing for test:
        # print("Shape of grid_temps2 after clamping:", grid_temps2.shape)

        for i in range(ds_length):
            img, gt = dataset[i]
            gt_type = type(gt)

            # # printing for testing:
            # print("Type of img:", type(img))
            # print("Shape of img:", img.shape)
            # if isinstance(dataset, torch.utils.data.Subset):
            #     print("Type of dataset.dataset.data[dataset.indices[i]]:", type(dataset.dataset.data[dataset.indices[i]]))
            #     print("Shape of dataset.dataset.data[dataset.indices[i]]:", dataset.dataset.data[dataset.indices[i]].shape)
            # elif isinstance(dataset, torch.utils.data.Dataset):
            #     print("Type of dataset.data[i]:", type(dataset.data[i]))
            #     print("Shape of dataset.data[i]:", dataset.data[i].shape)
            # else:
            #     print("Dataset is neither a torch.utils.data.Dataset nor a torch.utils.data.Subset")

            # noise image
            if ct < num_cross and cross_indices[ct] == i:
                cross_id.append(cnt)
                img = F.grid_sample(img.unsqueeze(0), grid_temps2, align_corners=True)[0]
                ct += 1

                # print("Shape of img after grid_sample (noise image):", img.shape)
                
            # poisoned image
            if pt < num_poison and poison_indices[pt] == i:
                poison_id.append(cnt)
                # change the label to the target class:
                tc = target_class if not all_to_all else (gt + 1) % num_classes
                if source_label is None:
                    gt = tc
                else:
                    gt = tc if int(gt) == int(source_label) else gt
                # apply the transformation to the image:
                img = F.grid_sample(img.unsqueeze(0), grid_temps, align_corners=True)[0]
                pt += 1

                # print("Shape of img after grid_sample (poisoned image):", img.shape)

            # img_file_name = '%d.png' % cnt
            # img_file_path = os.path.join(self.path, img_file_name)
            # save_image(img, img_file_path)
            # print('[Generate Poisoned Set] Save %s' % img_file_path)

            img_set.append(img.unsqueeze(0))
            label_set.append(gt_type(gt))
            cnt += 1

        img_set = torch.cat(img_set, dim=0)
        
        poison_indices = poison_id
        cross_indices = cross_id

    # if preparing testset:
    else:
        for i in range(ds_length):
            img, gt = dataset[i]
            gt_type = type(gt)
            # change the label to the target class
            tc = target_class if not all_to_all else (gt + 1) % num_classes
            if gt != tc:
                if source_label is None:
                    gt = tc
                else:
                    gt = tc if int(gt) == int(source_label) else gt

                img = F.grid_sample(img.unsqueeze(0), grid_temps, align_corners=True)[0]
                poison_id.append(i)
            img_set.append(img.unsqueeze(0))
            label_set.append(gt_type(gt))
        img_set = torch.cat(img_set, dim=0)
        poison_indices = poison_id
        


    return PoisonedDataset(img_set, label_set, poison_indices, cross_indices)



class PoisonedDataset(torch.utils.data.Dataset):
        def __init__(self, images, labels, poison_indices, cross_indices):
            self.images = images
            self.labels = labels
            self.poison_indices = poison_indices
            self.cross_indices = cross_indices

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            return self.images[idx], self.labels[idx]
        

# The following classes: ProbTransform, PostTensorTransform are taken from the wanet original code:
# https://github.com/VinAIResearch/Warping-based_Backdoor_Attack-release.git
# In the original code, they applied PostTensorTransform on images after poisoning the image and before feeding it to the model. 
# They used this just in training phase.

class ProbTransform(torch.nn.Module):
    def __init__(self, f, p=1):
        super(ProbTransform, self).__init__()
        self.f = f
        self.p = p

    def forward(self, x):  # , **kwargs):
        if random.random() < self.p:
            return self.f(x)
        else:
            return x

class PostTensorTransform(torch.nn.Module):

    """
    The PostTensorTransform class is a subclass of torch.nn.Module that applies a series of transformations to an input tensor.
    The transformations include random cropping, random rotation, and random horizontal flipping.

    Attributes:
        random_crop (ProbTransform): An instance of the ProbTransform class that applies random cropping to the input tensor.
        random_rotation (ProbTransform): An instance of the ProbTransform class that applies random rotation to the input tensor.
        random_horizontal_flip (kornia.augmentation.RandomHorizontalFlip): A method that applies random horizontal flipping to the input tensor.

    Args:
        opt (object): An object that contains the configuration parameters for the transformations. It should have the following attributes:
            input_height (int): The height of the input tensor.
            input_width (int): The width of the input tensor.
            random_crop (int): The padding size for the random crop transformation.
            random_rotation (float): The degree range for the random rotation transformation.
            dataname (str): The name of the dataset. If it is "cifar10" or "tinyimagenet", random horizontal flipping will be applied.
    """

    
    def __init__(self, opt):
        super(PostTensorTransform, self).__init__()
        self.random_crop = ProbTransform(
            A.RandomCrop((opt.input_height, opt.input_width), padding=opt.random_crop), p=0.8
        )
        self.random_rotation = ProbTransform(A.RandomRotation(opt.random_rotation), p=0.5)
        if opt.dataname in ["cifar10", 'tinyimagenet']:
            self.random_horizontal_flip = A.RandomHorizontalFlip(p=0.5)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x
    
