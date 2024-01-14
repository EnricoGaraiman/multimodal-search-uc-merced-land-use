from torch import nn

from src.CNN import CNN
import torchvision.models as models


def create_model(PARAMETERS):
    """
    Create model

    @param PARAMETERS: global parameters
    @type PARAMETERS: dict
    @return: model
    @rtype: object
    """
    if PARAMETERS['model'] == 'cnn':
        return create_cnn_model(PARAMETERS)
    elif PARAMETERS['model'] == 'resnet':
        return create_resnet_model(PARAMETERS)
    elif PARAMETERS['model'] == 'vgg':
        return create_vgg_model(PARAMETERS)


def create_cnn_model(PARAMETERS):
    """
    Create Custom CNN model

    @param PARAMETERS: global parameters
    @type PARAMETERS: dict
    @return: model
    @rtype: object
    """
    model = CNN(num_classes=PARAMETERS['num_classes'])

    return model


def create_resnet_model(PARAMETERS):
    """
    Create ResNet model

    @param PARAMETERS: global parameters
    @type PARAMETERS: dict
    @return: model
    @rtype: object
    """
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, PARAMETERS['num_classes'])

    return model


def create_vgg_model(PARAMETERS):
    """
    Create VGG model

    @param PARAMETERS: global parameters
    @type PARAMETERS: dict
    @return: model
    @rtype: object
    """
    model = models.vgg13_bn(weights=None, num_classes=PARAMETERS['num_classes'])

    return model
