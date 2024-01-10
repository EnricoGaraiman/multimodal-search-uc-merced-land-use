import os

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from skimage import io
import glob


def load_uc_merced_land_use_dataset(PARAMETERS, root, seed=42):
    torch.manual_seed(seed)

    transform = transforms.Compose(
        [transforms.Resize((PARAMETERS['size'], PARAMETERS['size'])), transforms.ToTensor(), ])

    dataset = datasets.ImageFolder(root=root, transform=transform)

    total_size = len(dataset)
    train_size = int(PARAMETERS['train_ratio'] * total_size)
    test_size = total_size - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size],
                                               generator=torch.Generator().manual_seed(seed))

    train_dataloader = DataLoader(train_dataset, batch_size=PARAMETERS['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=PARAMETERS['batch_size'], shuffle=False)

    classes = sorted(os.listdir(root))

    return train_dataset, test_dataset, train_dataloader, test_dataloader, classes


def dataset_examples_each_class(data_train_dir, show=True):
    fig = plt.figure(figsize=(18, 9))
    for index, class_dir in enumerate(glob.glob(data_train_dir + '/*')):
        plt.subplot(5, 5, index + 1)
        img = io.imread(glob.glob(class_dir + '/*')[4])
        plt.imshow(img, cmap='gray')
        plt.title((class_dir.split('\\')[-1]).split('-')[-1])

    plt.tight_layout()
    if show:
        plt.show()
    fig.savefig('results/training_data_visualisation_.jpg')


def dataset_distribution(data_train_dir):
    all_classes_directory = glob.glob(data_train_dir + '/*')
    class_images_distribution = []
    class_labels = []
    for path in all_classes_directory:
        class_images_distribution.append(len(glob.glob(path + '/*')))
        class_labels.append((path.split('\\')[-1]).split('-')[-1])

    x_pos = [i for i, _ in enumerate(class_labels)]

    fig, ax = plt.subplots(figsize=(18, 15))
    plt.bar(x_pos, class_images_distribution, color='green')
    plt.xlabel("Number of images", fontsize=16)
    plt.ylabel("Class", fontsize=16)
    plt.title("Data distribution", fontsize=16)
    plt.xticks(x_pos, class_labels, fontsize=16, rotation=45, ha='right')
    plt.tight_layout()
    fig.savefig('results/training_data_distribution.jpg')
