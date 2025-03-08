import os
import torch
import torchvision
import torchvision.transforms as transforms
from appfl.misc.data import (
    Dataset,
    iid_partition,
    class_noniid_partition,
    dirichlet_noniid_partition,
    fixed_partition
)


def get_cifar10(
    num_clients: int, client_id: int, partition_strategy: str = "iid", **kwargs
):
    """
    Return the CIFAR10 dataset for a given client.
    :param num_clients: total number of clients
    :param client_id: the client id
    """
    # Get the download directory for dataset
    dir = os.getcwd() + "/datasets/RawData"

    # Root download the data if not already available.
    test_data_raw = torchvision.datasets.CIFAR10(
        dir, download=True, train=False, transform=transforms.ToTensor()
    )

    # Obtain the testdataset
    test_data_input = []
    test_data_label = []
    for idx in range(len(test_data_raw)):
        test_data_input.append(test_data_raw[idx][0].tolist())
        test_data_label.append(test_data_raw[idx][1])
    test_dataset = Dataset(
        torch.FloatTensor(test_data_input), torch.tensor(test_data_label)
    )

    # Training data for multiple clients
    train_data_raw = torchvision.datasets.CIFAR10(
        dir, download=False, train=True, transform=transforms.ToTensor()
    )

    fixed_partition_table = {
        0: {0: 211, 1: 53, 2: 53, 3: 53, 4: 53, 5: 211, 6: 52, 7: 52, 8: 211, 9: 52},
        1: {0: 52, 1: 658, 2: 218, 3: 52, 4: 218, 5: 52, 6: 218, 7: 52, 8: 218, 9: 52},
        2: {0: 488, 1: 748, 2: 249, 3: 50, 4: 250, 5: 250, 6: 50, 7: 50, 8: 250, 9: 50},
        3: {0: 450, 1: 237, 2: 237, 3: 237, 4: 1000, 5: 50, 6: 50, 7: 237, 8: 237, 9: 238},
        4: {0: 386, 1: 194, 2: 48, 3: 578, 4: 194, 5: 386, 6: 200, 7: 1000, 8: 386, 9: 577},
        5: {0: 418, 1: 418, 2: 418, 3: 418, 4: 418, 5: 800, 6: 200, 7: 600, 8: 600, 9: 200},
        6: {0: 166, 1: 332, 2: 829, 3: 995, 4: 497, 5: 800, 6: 1000, 7: 600, 8: 400, 9: 200},
        7: {0: 324, 1: 974, 2: 1000, 3: 324, 4: 324, 5: 1200, 6: 400, 7: 400, 8: 800, 9: 1200},
        8: {0: 697, 1: 696, 2: 1220, 3: 175, 4: 697, 5: 800, 6: 1400, 7: 200, 8: 400, 9: 1000},
        9: {0: 1138, 1: 50, 2: 200, 3: 1400, 4: 800, 5: 400, 6: 1200, 7: 1400, 8: 1000, 9: 1200}
    }

    # Partition the dataset
    if partition_strategy == "iid":
        train_datasets = iid_partition(train_data_raw, num_clients)
    elif partition_strategy == "class_noniid":
        train_datasets = class_noniid_partition(train_data_raw, num_clients, **kwargs)
    elif partition_strategy == "dirichlet_nomiid":
        train_datasets = dirichlet_noniid_partition(
            train_data_raw, num_clients, **kwargs  # TODO adjusted the alpha parameters
        )
    elif partition_strategy == "fixed":
        train_datasets = fixed_partition(
            train_data_raw,
            num_clients,
            fixed_partition_table,
            **kwargs,
        )
    else:
        raise ValueError(f"Invalid partition strategy: {partition_strategy}")

    return train_datasets[client_id], test_dataset
