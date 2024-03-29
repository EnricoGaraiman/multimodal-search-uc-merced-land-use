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
        'epochs': 100,
        'model': 'resnet',
        'early_stopping_epochs_no_best': 20,
        'size': 224,
        'learning_rate': 1e-3,
        'train_ratio': 0.8,
        'num_classes': 21,
        'batch_size': 10,
        'database': 'data/database_resnet34_imagenet.db',
        'dataset_path': 'data/UCMerced_LandUse/Images'
    }

    images_to_search = [
        'data/testing/airplane2-test.jpg',
        'data/testing/buildings-test.jpg',
        'data/testing/forest-test.jpg',
        'data/testing/intersection-test.png',
        'data/testing/river-test2.jpg',
        'data/testing/tennis-test.jpg',
    ]
    texts_to_search = [
        'I need a image with an airplane',
        'I want a river photo',
        'I want to play tennis',
        'I like forest',
        'My car is stuck in an intersection',
        'This city is full with buildings',
    ]

    # dataset
    dataset.dataset_examples_each_class(PARAMETERS['dataset_path'], False)
    dataset.dataset_distribution(PARAMETERS['dataset_path'])

    # train_dataset, test_dataset, dataset_train_loader, dataset_test_loader, classes = dataset.load_uc_merced_land_use_dataset(
    #     PARAMETERS, root=PARAMETERS['dataset_path'])
    #
    # # training
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
    # model, predictions, real_labels = train.plot_confusion_matrix(PARAMETERS,
    #                                                               dataset_test_loader=dataset_test_loader,
    #                                                               best_model=best_model,
    #                                                               classes=classes)
    #
    # # search
    # features_model = torch.nn.Sequential(*(list(model.children())[:-1]))
    #
    # search.insert_all_images_from_database(PARAMETERS, features_model)
    #
    # if len(images_to_search) > 0:
    #     for image_to_search in images_to_search:
    #         search.search(PARAMETERS, features_model, image_to_search)
    #
    # if len(texts_to_search):
    #     for text_to_search in texts_to_search:
    #         search.search_by_text(PARAMETERS, text_to_search)
