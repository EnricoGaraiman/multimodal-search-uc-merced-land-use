from src.CNN import CNN
import torch
import torchvision.models as models


def create_model(PARAMETERS):
    if PARAMETERS['model'] == 'cnn':
        return create_cnn_model(PARAMETERS)
    elif PARAMETERS['model'] == 'resnet':
        return create_resnet_model(PARAMETERS)


def create_cnn_model(PARAMETERS):
    model = CNN(num_classes=PARAMETERS['num_classes'])

    return model


def create_resnet_model(PARAMETERS):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, PARAMETERS['num_classes'])

    return model
