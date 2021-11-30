# import Torch packages and its submodules
import torch
from torch import nn, optim

# import TorchVision and its submodules
import torch.nn.functional as F
from torchvision import datasets, transforms

# import other packages
import numpy as np
import math
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import shutil, os

RAW_DIR_PATH = "./notMNIST_small"
PATH = "./splitted_imagesets"


def _remove_empty_images(list, path):
    """
    Remove all the images that has zero bytes in size.
    :param list: the list of names of files in the directory
    :param path: the path of the directory
    :return: the new list with empty images removed
    """
    new_list = []
    for item in list:
        if os.path.getsize(f"{path}/{item}") > 0:
            new_list.append(item)
        else:
            print(f"An image is deleted: {item} from {path}")
    return new_list


def step_0_split_data():
    """
    Select 1500 photos for training, 100 photos for validation and the rest for
    the testing per each letter directory in the notMNIST data.
    :return: None
    """
    letters = os.listdir(RAW_DIR_PATH)
    labels = ["train", "test", "validation"]
    if not os.path.isdir(PATH):
        os.mkdir(PATH)
    for label in labels:
        os.mkdir(os.path.join(PATH, label))
    for letter in letters:
        source = os.path.join(RAW_DIR_PATH, letter)
        images = _remove_empty_images(os.listdir(source), source)
        # Train - 1500
        images_trained = np.random.choice(images, size=1500, replace=False)
        # Validation - 100
        images = list(set(images).difference(images_trained))
        images_validated = np.random.choice(images, size=100, replace=False)
        # Test - remaining images
        images_test = list(set(images).difference(images_validated))
        images_letter = [images_trained, images_test, images_validated]
        for i in range(len(images_letter)):
            destination = os.path.join(PATH, os.path.join(labels[i], letter))
            os.mkdir(destination)
            for image in images_letter[i]:
                shutil.copy(os.path.join(source, image), destination)
            print(f"Letter {letter} {labels[i]} folder copy complete")

TRAIN_DATA_PATH = "./splitted_imagesets/train"
TEST_DATA_PATH = "./splitted_imagesets/test"
VALID_DATA_PATH = "./splitted_imagesets/validation"


def step_1_load_data(gaussian=False, normalized=True):
    """
    Preprocessed data and load the data such that trainloaders, validloaders
    and testloaders can be created for the rest of the codes to use.
    :param gaussian: Boolean to indicates whether we want gaussian blur or not
    in the data preprocessing stage
    :return:
    1) trainloader: loader to load training data
    2) validloader: Loader to load validation data
    3) testloader: loader to load testing data
    """
    # Define a transform to normalize the data
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])
    if gaussian:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.GaussianBlur(3, sigma=0.7),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])
    if not normalized:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()])
    trainset = datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True)

    testset = datasets.ImageFolder(root=TEST_DATA_PATH, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                             shuffle=True)

    validset = datasets.ImageFolder(root=VALID_DATA_PATH, transform=transform)
    validloader = torch.utils.data.DataLoader(validset, batch_size=64,
                                              shuffle=True)
    return trainloader, testloader, validloader


def _build_neural_network(hidden_units):
    input_size = 784
    output_size = 10
    model = nn.Sequential(nn.Linear(in_features=input_size, out_features=hidden_units),
                          nn.ReLU(),
                          nn.Linear(in_features=hidden_units, out_features=output_size),
                          nn.Softmax(dim=1))
    return model


def _train_validate_and_test_model(hidden_units, optimizer_n, lr, trainloader, validloader, testloader, test=True, early_stopping=3, epochs=10, two_layers=False, dropout=False, suffix=""):
    """
    Train the validates the model. For task 2, also tested the model in each epoch.
    :param hidden_units: the number of hidden units
    :param optimizer_n: the name of the optimizer
    :param lr: the learning rate that we are going to use
    :param trainloader: the loader to load training data
    :param validloader: the loader to load validation data
    :param testloader: the lodaer to load testing data
    :param test: whether we want to include testing in each epoch or not
    :param early_stopping: the number of loops to observe for no improvement before early stopping
    :param epochs: the maximum number of epochs
    :param two_layers: whether we are running a two layer model or not
    :param dropout: whether we are running a model with dropout or not
    :param suffix: the suffix that we want for the model name
    :return: DataFrame with all the training, validation and testing losses and accuracies
    """
    if two_layers:
        model = _build_2_layers_network(hidden_units)
    else:
        model = _build_neural_network(hidden_units)
    if dropout:
        model = _build_network_with_dropout(hidden_units)
    criterion = nn.CrossEntropyLoss()
    valid_loss_min = np.Inf
    epochs_no_improve, valid_best_acc = 0, 0
    if not os.path.isdir("./models"):
        os.mkdir("./models")
    save_file_name = f"models/model-{optimizer_n}-{lr}-{hidden_units}{suffix}.pt"
    # Use SGD optimizer for now
    if optimizer_n == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_n == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_n == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    else:
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
    outputs_losses = []
    losses_colums = ["training_loss","validation_loss"]
    outputs_acc = []
    acc_columns = ["training_acc","validation_acc"]
    if test:
        losses_colums.append("testing_loss")
        acc_columns.append("testing_acc")
    for e in range(epochs):
        training_loss, validation_loss, testing_loss = 0, 0, 0
        training_acc, validation_acc, testing_acc = 0, 0, 0
        for images, labels in trainloader:
            images = images.view(images.shape[0], -1)
            optimizer.zero_grad()
            print(images.shape)
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            training_loss += loss.item() * images.size(0)
            # Get training accuracy
            _, pred = torch.max(output, dim=1)
            correct_tensor = pred.eq(labels.data.view_as(pred))
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            training_acc += accuracy.item() * images.size(0)
        else:
            with torch.no_grad():
                model.eval()
                print("model start validating")
                for images, labels in validloader:
                    images = images.view(images.shape[0], -1)
                    output = model(images)
                    loss = criterion(output, labels)
                    validation_loss += loss.item() * images.size(0)
                    # Get Validation accuracy
                    _, pred = torch.max(output, dim=1)
                    correct_tensor = pred.eq(labels.data.view_as(pred))
                    accuracy = torch.mean(
                        correct_tensor.type(torch.FloatTensor))
                    validation_acc += accuracy.item() * images.size(0)
                print("model start testing")
                if test:
                    for images, labels in testloader:
                        images = images.view(images.shape[0], -1)
                        output = model(images)
                        loss = criterion(output, labels)
                        testing_loss += loss.item() * images.size(0)
                        _, pred = torch.max(output, dim=1)
                        correct_tensor = pred.eq(labels.data.view_as(pred))
                        accuracy = torch.mean(
                            correct_tensor.type(torch.FloatTensor)
                        )
                        testing_acc += accuracy.item() * images.size(0)


                # Calculate average losses
                train_loss = training_loss / len(trainloader.dataset)
                valid_loss = validation_loss / len(validloader.dataset)
                test_loss = testing_loss / len(testloader.dataset)
                # Calculate average accuracy
                train_acc = training_acc / len(trainloader.dataset)
                valid_acc = validation_acc / len(validloader.dataset)
                test_acc = testing_acc / len(testloader.dataset)

                if test:
                    outputs_losses.append([
                        train_loss, valid_loss, test_loss
                    ])
                    outputs_acc.append([
                        train_acc, valid_acc, test_acc
                    ])
                else:
                    outputs_losses.append([
                        train_loss, valid_loss
                    ])
                    outputs_acc.append([
                        train_acc, valid_acc
                    ])

                # Save the model if validation loss decreases
                if valid_loss < valid_loss_min:
                    # Save model
                    torch.save(model.state_dict(), save_file_name)
                    # Track improvement
                    epochs_no_improve = 0
                    valid_loss_min = valid_loss

                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= early_stopping:
                        model.load_state_dict(torch.load(save_file_name))
                        model.optimizer = optimizer
                        outputs_losses = pd.DataFrame(
                            outputs_losses,
                            columns=losses_colums
                        )
                        outputs_acc = pd.DataFrame(
                            outputs_acc,
                            columns=acc_columns
                        )
                        return model, outputs_losses, outputs_acc

        print(f"Epoch {e} complete")

    model.optimizer = optimizer
    outputs_losses = pd.DataFrame(
        outputs_losses,
        columns=losses_colums
    )
    outputs_acc = pd.DataFrame(
        outputs_acc,
        columns=acc_columns
    )
    return model, outputs_losses, outputs_acc


def step_2_get_optimal_learning_rate(trainloader, validloader, testloader):
    """
    Generates model results for five different learning rates and 4 different
    optimizer.
    :param trainloader: loader to laod training data
    :param validloader: loader to load validation data
    :param testloader: loader to load testing data
    :return:
    """
    hidden_unit = 1000
    lrs = [0.001, 0.01, 0.05, 0.1, 0.5]
    n_epochs = 15
    optimizer_ns = ["Adadelta", "Adam", "RMSProp", "SGD"]
    if not os.path.isdir("./DataResults"):
        os.mkdir("./DataResults")
    if not os.path.isdir("./plots"):
        os.mkdir("./plots")
    for optimizer_n in optimizer_ns:
        for lr in lrs:
            _, losses, acc = _train_validate_and_test_model(hidden_unit, optimizer_n, lr, trainloader, validloader, testloader, epochs=n_epochs)
            print(losses, acc)
            result = pd.concat([losses, acc], axis=1)
            result.to_csv(f"./DataResults/step_2_{optimizer_n}_lr={lr}_epochs={n_epochs}_hidden={hidden_unit}.csv")
            lines = losses.plot.line()
            fig = lines.get_figure()
            fig.savefig(f"./plots/step_2_{optimizer_n}_lr={lr}_epochs={n_epochs}_hidden={hidden_unit}_LOSS.png")
            accuracies = acc.plot.line()
            fig = accuracies.get_figure()
            fig.savefig(
                f"./plots/step_2_{optimizer_n}_lr={lr}_epochs={n_epochs}_hidden={hidden_unit}_ACC.png")


def test_model_independently(testloader, model_name, hidden_unit, two_layers=False, dropout=False):
    if two_layers:
        model = _build_2_layers_network(hidden_unit)
    else:
        model = _build_neural_network(hidden_unit)
    if dropout:
        model = _build_network_with_dropout(hidden_unit)
    model.load_state_dict(torch.load(model_name))
    criterion = nn.CrossEntropyLoss()
    testing_acc, testing_loss = 0, 0
    outputs = []
    with torch.no_grad():
        for images, labels in testloader:
            images = images.view(images.shape[0], -1)
            output = model(images)
            loss = criterion(output, labels)
            testing_loss += loss.item() * images.size(0)
            _, pred = torch.max(output, dim=1)
            correct_tensor = pred.eq(labels.data.view_as(pred))
            accuracy = torch.mean(
                correct_tensor.type(torch.FloatTensor)
            )
            testing_acc += accuracy.item() * images.size(0)
        test_loss = testing_loss / len(testloader.dataset)
        test_acc = testing_acc / len(testloader.dataset)
        outputs.append([test_loss, test_acc])
    return pd.DataFrame(outputs, columns=["test_loss", "test_acc"])


def step_3_get_optimal_hidden_units(trainloader, validloader, testloader):
    hidden_units = [100, 500, 1000]
    lrs = [0.5]
    optimizer_n = "RMSProp"
    n_epochs = 15
    if not os.path.isdir("./DataResults"):
        os.mkdir("./DataResults")
    if not os.path.isdir("./plots"):
        os.mkdir("./plots")
    record = []
    min_valid_losses = float("inf")
    min_index = 0
    for hidden_unit in hidden_units:
        for lr in lrs:
            _, losses, acc = _train_validate_and_test_model(hidden_unit,
                                                            optimizer_n, lr,
                                                            trainloader,
                                                            validloader, testloader,
                                                            epochs=n_epochs,
                                                            test=False,
                                                            suffix="step3")
            print(losses, acc)
            min_valid_losses = min(min_valid_losses, losses["validation_loss"].iloc[-1])
            record.append((hidden_unit, lr, losses["validation_loss"].iloc[-1]))
            if min_valid_losses < min(record)[2]:
                min_index = len(record)-1
            result = pd.concat([losses, acc], axis=1)
            result.to_csv(
                f"DataResults/step_3_{optimizer_n}_lr={lr}_epochs={n_epochs}_hidden={hidden_unit}.csv")
            lines = losses.plot.line()
            fig = lines.get_figure()
            fig.savefig(
                f"./plots/step_3_{optimizer_n}_lr={lr}_epochs={n_epochs}_hidden={hidden_unit}_LOSS.png")
            accuracies = acc.plot.line()
            fig = accuracies.get_figure()
            fig.savefig(
                f"./plots/step_3_{optimizer_n}_lr={lr}_epochs={n_epochs}_hidden={hidden_unit}_ACC.png")
    hidden_unit, lr, loss = record[min_index]
    print(loss)
    model_name = f"models/model-{optimizer_n}-{lr}-{hidden_unit}step3.pt"
    test_outputs = test_model_independently(testloader, model_name=model_name, hidden_unit=hidden_unit)
    test_outputs.to_csv(
        f"DataResults/step_3_test_{optimizer_n}_lr={lr}_epochs={n_epochs}_hidden={hidden_unit}.csv")


def _build_2_layers_network(hidden_units):
    input_size = 784
    output_size = 10
    hidden_size = hidden_units
    model = nn.Sequential(nn.Linear(input_size, hidden_size),
                          nn.ReLU(),
                          nn.Linear(hidden_size, hidden_size),
                          nn.ReLU(),
                          nn.Linear(hidden_size, output_size),
                          nn.Softmax(dim=1))
    return model


def step_4_test_2_layers(trainloader, validloader, testloader):
    lrs = [0.5]
    optimizer_n = "RMSProp"
    n_epochs = 15
    hidden_unit = 500
    if not os.path.isdir("./DataResults"):
        os.mkdir("./DataResults")
    if not os.path.isdir("./plots"):
        os.mkdir("./plots")
    for lr in lrs:
        _, losses, acc = _train_validate_and_test_model(hidden_unit,
                                                        optimizer_n, lr,
                                                        trainloader,
                                                        validloader, testloader,
                                                        epochs=n_epochs,
                                                        two_layers=True,
                                                        test=False,
                                                        suffix="twolayers")
        _, losses1, acc1 = _train_validate_and_test_model(1000,
                                                        optimizer_n, lr,
                                                        trainloader,
                                                        validloader, testloader,
                                                        epochs=n_epochs,
                                                        test=False,
                                                        suffix="onelayers")
        result = pd.concat([losses, acc], axis=1)
        result2 = pd.concat([losses1, acc1], axis=1)
        result.to_csv(
            f"DataResults/step_4_two_layers_{optimizer_n}_lr={lr}_epochs={n_epochs}_hidden={hidden_unit}.csv")
        result2.to_csv(
            f"DataResults/step_4_{optimizer_n}_lr={lr}_epochs={n_epochs}_hidden={1000}.csv")
        lines = losses.plot.line()
        fig = lines.get_figure()
        fig.savefig(
            f"./plots/step_4_two_layer_{optimizer_n}_lr={lr}_epochs={n_epochs}_hidden={hidden_unit}_LOSS.png")
        accuracies = acc.plot.line()
        fig = accuracies.get_figure()
        fig.savefig(
            f"./plots/step_4_two_layer_{optimizer_n}_lr={lr}_epochs={n_epochs}_hidden={hidden_unit}_ACC.png")
        lines = losses1.plot.line()
        fig = lines.get_figure()
        fig.savefig(
            f"./plots/step_4_{optimizer_n}_lr={lr}_epochs={n_epochs}_hidden={1000}_LOSS.png")
        accuracies = acc1.plot.line()
        fig = accuracies.get_figure()
        fig.savefig(
            f"./plots/step_4_{optimizer_n}_lr={lr}_epochs={n_epochs}_hidden={1000}_ACC.png")
    model_name = f"models/model-{optimizer_n}-{lrs[0]}-{hidden_unit}twolayers.pt"
    test_outputs = test_model_independently(testloader, model_name=model_name,
                                            hidden_unit=hidden_unit, two_layers=True)
    test_outputs.to_csv(
        f"DataResults/step_4_test_{optimizer_n}_lr={lrs[0]}_epochs={n_epochs}_hidden={hidden_unit}twolayers.csv")
    model_name = f"models/model-{optimizer_n}-{lrs[0]}-{1000}onelayers.pt"
    test_outputs = test_model_independently(testloader, model_name=model_name,
                                            hidden_unit=1000)
    test_outputs.to_csv(
        f"DataResults/step_4_test_{optimizer_n}_lr={lrs[0]}_epochs={n_epochs}_hidden={1000}onelayers.csv")


def _build_network_with_dropout(hidden_units):
    input_size = 784
    output_size = 10
    model = nn.Sequential(
        nn.Linear(in_features=input_size, out_features=hidden_units),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=hidden_units, out_features=output_size),
        nn.Softmax(dim=1))
    return model


def step_5_test_dropout(trainloader, testloader, validloader):
    lrs = [0.5]
    optimizer_n = "RMSProp"
    n_epochs = 3
    hidden_unit = 1000
    if not os.path.isdir("./DataResults"):
        os.mkdir("./DataResults")
    if not os.path.isdir("./plots"):
        os.mkdir("./plots")
    for lr in lrs:
        _, losses, acc = _train_validate_and_test_model(hidden_unit,
                                                        optimizer_n, lr,
                                                        trainloader,
                                                        validloader, testloader,
                                                        epochs=n_epochs,
                                                        dropout=True,
                                                        test=False,
                                                        suffix="dropout")
        print(losses, acc)
        result = pd.concat([losses, acc], axis=1)
        result.to_csv(
            f"DataResults/step_5_{optimizer_n}_lr={lr}_epochs={n_epochs}_hidden={hidden_unit}.csv")
        lines = losses.plot.line()
        fig = lines.get_figure()
        fig.savefig(
            f"./plots/step_5_{optimizer_n}_lr={lr}_epochs={n_epochs}_hidden={hidden_unit}_LOSS.png")
        accuracies = acc.plot.line()
        fig = accuracies.get_figure()
        fig.savefig(
            f"./plots/step_5_{optimizer_n}_lr={lr}_epochs={n_epochs}_hidden={hidden_unit}_ACC.png")
    model_name = f"models/model-{optimizer_n}-{lrs[0]}-{hidden_unit}dropout.pt"
    test_outputs = test_model_independently(testloader, model_name=model_name,
                                            hidden_unit=hidden_unit, dropout=True)
    test_outputs.to_csv(
        f"DataResults/step_5_test_{optimizer_n}_lr={lrs[0]}_epochs={n_epochs}_hidden={hidden_unit}onelayers.csv")


def test_gaussian_blur(trainloader,validloader, testloader):
    lr = 0.5
    optimizer_n = "RMSProp"
    n_epochs = 15
    hidden_unit = 1000
    _, losses, acc = _train_validate_and_test_model(hidden_unit, optimizer_n,
                                                    lr, trainloader,
                                                    validloader, testloader,
                                                    epochs=n_epochs,
                                                    test=False,
                                                    suffix="gaussian")
    print(losses, acc)
    result = pd.concat([losses, acc], axis=1)
    result.to_csv(
        f"DataResults/gaussian_{optimizer_n}_lr={lr}_epochs={n_epochs}_hidden={hidden_unit}.csv")
    lines = losses.plot.line()
    fig = lines.get_figure()
    fig.savefig(
        f"./plots/gaussian_{optimizer_n}_lr={lr}_epochs={n_epochs}_hidden={hidden_unit}_LOSS.png")
    accuracies = acc.plot.line()
    fig = accuracies.get_figure()
    fig.savefig(
        f"./plots/gaussian_{optimizer_n}_lr={lr}_epochs={n_epochs}_hidden={hidden_unit}_ACC.png")


def test_normalize(trainloader,validloader, testloader):
    lr = 0.5
    optimizer_n = "RMSProp"
    n_epochs = 15
    hidden_unit = 1000
    _, losses, acc = _train_validate_and_test_model(hidden_unit, optimizer_n,
                                                    lr, trainloader,
                                                    validloader, testloader,
                                                    epochs=n_epochs,
                                                    test=False,
                                                    suffix="nonormalized")
    print(losses, acc)
    result = pd.concat([losses, acc], axis=1)
    result.to_csv(
        f"DataResults/Nonormalized_{optimizer_n}_lr={lr}_epochs={n_epochs}_hidden={hidden_unit}.csv")
    lines = losses.plot.line()
    fig = lines.get_figure()
    fig.savefig(
        f"./plots/Nonormalized_{optimizer_n}_lr={lr}_epochs={n_epochs}_hidden={hidden_unit}_LOSS.png")
    accuracies = acc.plot.line()
    fig = accuracies.get_figure()
    fig.savefig(
        f"./plots/Nonormalized_{optimizer_n}_lr={lr}_epochs={n_epochs}_hidden={hidden_unit}_ACC.png")


# if __name__ == "__main__":
    # =============== Part 1 - Data Preprocessing ===============

    # Split data in `nonMNIST_small` into training, validation and testing set
    # Before running line 558, please make sure that the splitted_imagesets
    # directory is deleted.
    # step_0_split_data()

    # Get the loaders for the training set, test set, and the validation set
    # trainloader, testloader, validloader = step_1_load_data()

    # ----------- Test Gaussian Blur filter ---------------
    # trainloader_gauss, testloader_gauss, validloader_gauss = step_1_load_data(gaussian=True)
    # test_gaussian_blur(trainloader_gauss, validloader_gauss, testloader_gauss)

    # ----------- Test Normalization filter ---------------
    # trainloader_nonorm, testloader_nonorm, validloader_nonorm = step_1_load_data(
    #     normalized=False)
    # test_normalize(trainloader_nonorm, validloader_nonorm, testloader_nonorm)


    # ============ Part two - Testing Learning rates and optimizers =========
    # trainloader, testloader, validloader = step_1_load_data()
    # step_2_get_optimal_learning_rate(trainloader, validloader, testloader)


    # ============ Part three - Testing hidden units =========
    # trainloader, testloader, validloader = step_1_load_data()
    # step_3_get_optimal_hidden_units(trainloader, validloader, testloader)


    # ============ Part four - Testing two layers =========
    # trainloader, testloader, validloader = step_1_load_data()
    # step_4_test_2_layers(trainloader, validloader, testloader)


    # ============ Part five - Testing dropout layer =========
    # trainloader, testloader, validloader = step_1_load_data()
    # step_5_test_dropout(trainloader, validloader, testloader)




