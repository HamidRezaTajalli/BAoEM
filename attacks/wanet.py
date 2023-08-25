import torch
import random
import torch.nn.functional as F


# TODO 2 ta tabe pas minevisam: yeki collan train ya test set ro barmigardune (be nazaram as backdoor toolbox estefadeh kon! vali kode Xing ro ham farda check kon
# TODO yeki ham eine transform tu backdoor toolbox amal mikoneh! yani dune dune daade haro mogheh train ya test behesh midi transform mikone barmigardoone!


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


def get_poisoned_dataset(is_train: bool, dataset: torch.utils.data.Dataset, sample_dimension: tuple, s: float, k: int,
                         grid_rescale: int, poison_rate: float, cross_ratio: float, noise_grid, identity_grid,
                         target_class: int, num_classes, all_to_all: bool, source_label: int = None):
    """
    :param is_train: if True, returns the train set, otherwise returns the test set
    :param dataset: the dataset to be poisoned
    :param sample_dimension: the dimension of the samples in the dataset
    :return: the poisoned dataset
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
    poison_indices, cover_indices = None, None

    grid_temps = (identity_grid + s * noise_grid / input_height) * grid_rescale
    grid_temps = torch.clamp(grid_temps, -1, 1)

    #printing for test:
    print(grid_temps.shape)
    print(noise_grid.shape)
    print(identity_grid.shape)

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
        print(grid_temps2.shape)

        for i in range(ds_length):
            img, gt = dataset[i]

            # printing for testing:
            print(type(img))
            print(img.shape)
            print(type(dataset.data[i]))
            print(dataset.data[i].shape)


            # noise image
            if ct < num_cross and cross_indices[ct] == i:
                cross_id.append(cnt)
                img = F.grid_sample(img.unsqueeze(0), grid_temps2, align_corners=True)[0]
                ## TODO: check how this is applied on 1 channel datasets like mnist. check Jing code!
                ct += 1

                print(img.shape)

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

                print(img.shape)

            # img_file_name = '%d.png' % cnt
            # img_file_path = os.path.join(self.path, img_file_name)
            # save_image(img, img_file_path)
            # print('[Generate Poisoned Set] Save %s' % img_file_path)

            img_set.append(img.unsqueeze(0))
            label_set.append(gt)
            cnt += 1

        img_set = torch.cat(img_set, dim=0)
        label_set = torch.LongTensor(label_set)
        poison_indices = poison_id
        cover_indices = cross_id
        print("Poison indices:", poison_indices)
        print("Cover indices:", cover_indices)
        # TODO kollan type va shape dataset va har input ro ghabl va baad print begir bebin dare chi kar mikoneh. khosusan 4 khate bala.

    # if preparing testset:
    else:
        for i in range(ds_length):
            img, gt = dataset[i]
            # change the label to the target class
            tc = target_class if not all_to_all else (gt + 1) % num_classes
            if gt != tc:
                if source_label is None:
                    gt = tc
                else:
                    gt = tc if int(gt) == int(source_label) else gt

                img = F.grid_sample(img.unsqueeze(0), grid_temps, align_corners=True)[0]
            img_set.append(img.unsqueeze(0))
            label_set.append(gt)
        img_set = torch.cat(img_set, dim=0)
        label_set = torch.LongTensor(label_set)

    return img_set, label_set, poison_indices, cover_indices
    # TODO: bayad in dataset haye return shode ro ye Dataset class dorost koni va be una tabdil koni ghable return.
    #  hala soale asli ine ke: aya tuye Datasete jadid transformation mikhaim? Aya tuye dataset Avvaliey bayad
    #  transformation mizadim? Aya moghe transform kardane khode image bayad rushun taghiir bedim ya na?
