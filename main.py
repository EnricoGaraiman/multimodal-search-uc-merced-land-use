import torch
import src.dataset as dataset
import src.train as train
import src.search as search

"""
   MAIN FUNCTION
"""
if __name__ == '__main__':
    # parameters
    PARAMETERS = {
        'epochs': 50,
        'model': 'resnet',
        'early_stopping_epochs_no_best': 20,
        'size': 224,
        'learning_rate': 1e-4,
        'train_ratio': 0.8,
        'num_classes': 21,
        'batch_size': 10,
        'database': 'data/database_resnet34.db',
        'dataset_path': 'data/UCMerced_LandUse/Images'
    }

    image_to_search = 'data/UCMerced_LandUse/Images/forest/forest11.tif'
    text_to_search = 'I need a image with an airplane'

    # dataset
    dataset.dataset_examples_each_class(PARAMETERS['dataset_path'], False)
    dataset.dataset_distribution(PARAMETERS['dataset_path'])

    train_dataset, test_dataset, dataset_train_loader, dataset_test_loader, classes = dataset.load_uc_merced_land_use_dataset(
        PARAMETERS, root=PARAMETERS['dataset_path'])

    # training
    # train_loss_history, train_acc_history, test_loss_history, test_acc_history, best_model = train.training_stage(
    #     PARAMETERS,
    #     dataset_train_loader=dataset_train_loader,
    #     dataset_test_loader=dataset_test_loader)
    #
    # # testing
    # train.plot_training_results(train_loss_history=train_loss_history,
    #                             train_acc_history=train_acc_history,
    #                             test_loss_history=test_loss_history,
    #                             test_acc_history=test_acc_history)
    model, predictions, real_labels = train.plot_confusion_matrix(PARAMETERS,
                                                                  dataset_test_loader=dataset_test_loader,
                                                                  best_model='results/finals/resnet34/best_80.pth',
                                                                  classes=classes)

    # search
    features_model = torch.nn.Sequential(*(list(model.children())[:-1]))

    # search.insert_all_images_from_database(PARAMETERS, features_model)

    if image_to_search:
        search.search(PARAMETERS, features_model, image_to_search)

    if text_to_search:
        search.search_by_text(PARAMETERS, text_to_search)
