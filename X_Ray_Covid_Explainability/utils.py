import PIL
import numpy as np
import os
import pandas as pd
import shutil
import pandas as pd
import os
import torch
import numpy as np
import torchvision
from torchvision import transforms
import time
import copy
from typing import Tuple
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from lime import lime_image
from skimage.segmentation import mark_boundaries
import torch.nn.functional as F


def lime_viz(
    image_pil: PIL.Image.Image,
    model: torch.nn.Module,
    device: torch.device,
    class_names: list,
    num_samples: int = 200,
    pil_resize_transform: torchvision.transforms = transforms.Compose([
        transforms.Resize((256, 256)),
    ]),
    data_transforms: torchvision.transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
) -> Tuple[np.ndarray, torch.Tensor]:
    """[Return LIME numpy tensor for visualization purposses and prediction of class from given model]

    Args:
        image_pil (PIL.Image.Image): [Image from data in PIL format]
        model (torch.nn.Module): [Pytorch full model]
        device (torch.device): [Cuda or CPU]
        class_names (list): [Name of classes of data]
        num_samples (int, optional): [Size of the neighborhood to learn the linear model]. Defaults to 200.
        pil_resize_transform (torchvision.transforms, optional): [Resize transformation for data]. Defaults to transforms.Compose([ transforms.Resize((256, 256)), ]).
        data_transforms (torchvision.transforms, optional): [Transformations data was trained with]. Defaults to transforms.Compose([ transforms.ToTensor(), transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] ) ]).

    Returns:
        Tuple[np.ndarray, torch.Tensor]: [description]
    """
    def batch_predict(images: np.ndarray) -> np.ndarray:
        """Pass a batch of numpy array images through a PyTorch neural network and return the model's predictions.

        Args:
            images (np.ndarray): A batch of numpy array images. Has the dimensions B x H x W x C

        Returns:
            np.ndarray: The predictions from the model for each image in `images`. This is an array with dimensions
                        (number of images in batch) x (number of classes).
        """
        model.eval()

        images = tuple(data_transforms(i) for i in images)
        batch = torch.stack(images, dim=0).float()

        model.to(device)
        batch = batch.to(device)

        logits = model(batch)
        probabilities = F.softmax(logits, dim=1)

        return probabilities.detach().cpu().numpy()

    image_pil = pil_resize_transform(image_pil)

    preprocessed_image = data_transforms(image_pil).unsqueeze(0)
    preprocessed_image = preprocessed_image.to(device)

    explainer = lime_image.LimeImageExplainer()

    # LIME Explanation.
    explanation = explainer.explain_instance(
        np.array(image_pil),
        batch_predict,
        top_labels=1,
        hide_color=0,
        num_samples=num_samples,
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=False,
        num_features=15,
        hide_rest=False,
    )
    img_boundry = mark_boundaries(temp / 255.0, mask)
    prediction = class_names[model(preprocessed_image).argmax().item()]

    return img_boundry, prediction


def gradcam_viz(
    model: torch.nn.Module,
    target_layer: torch.nn.modules.conv,
    image_pil: torch.Tensor,
    label_tensor: torch.Tensor, device: torch.device,
    data_transforms=transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]),
    mean: list = [0.485, 0.456, 0.406],
    std: list = [0.229, 0.224, 0.225]
) -> Tuple[np.ndarray, torch.Tensor]:
    """[Return GradCam numpy tensor for visualization purposses and prediction of class from given model]

    Args:
        model (torch.nn.Module): [Pytorch full model]
        target_layer (torch.nn.modules.conv): [Last convolution layer in model]
        image_pil (torch.Tensor): [Image in PIL format from dataset]
        label_tensor (torch.Tensor): [Label tensor from dataset]
        device (torch.device): [Cuda or CPU]
        data_transforms ([type], optional): [Resize and tensor transforms]. Defaults to transforms.Compose([ transforms.Resize((256, 256)), transforms.ToTensor(), ]).
        mean (list, optional): [Mean for data normalization]. Defaults to [0.485, 0.456, 0.406].
        std (list, optional): [Std for data normalization]. Defaults to [0.229, 0.224, 0.225].

    Returns:
        Tuple[np.ndarray, torch.Tensor]: [description]
    """

    image_tensor = data_transforms(image_pil)
    # We convert the unnormalized image to a numpy array so we can use it to vizualize GradCAM.
    image_array = image_tensor.numpy().transpose(1, 2, 0)

    image_tensor_normalized = transforms.Normalize(mean=mean, std=std)(
        # To pass an image image through a model it must have dimensions BxCxHxW
        image_tensor.unsqueeze(0)
    )
    image_tensor_normalized = image_tensor_normalized.to(device)

    prediction = model(image_tensor_normalized)

    cam = GradCAM(
        model=model,
        target_layers=[target_layer],
        use_cuda=True if torch.cuda.is_available() else False
    )

    # Generate the GradCam heatmap array.
    grayscale_cam = cam(input_tensor=image_tensor_normalized,
                        target_category=label_tensor)
    # Blend the GradCam heatmap and unnormalized image.
    visualization = show_cam_on_image(
        image_array, grayscale_cam[0], use_rgb=True)

    _, preds = torch.max(prediction, 1)

    return visualization, preds


class F1Score:
    """
    Class for f1 calculation in Pytorch.
    """

    def __init__(self, average: str = 'weighted'):
        """
        Init.

        Args:
            average: averaging method
        """
        self.average = average
        if average not in [None, 'micro', 'macro', 'weighted']:
            raise ValueError('Wrong value of average parameter')

    @staticmethod
    def calc_f1_micro(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate f1 micro.

        Args:
            predictions: tensor with predictions
            labels: tensor with original labels

        Returns:
            f1 score
        """
        true_positive = torch.eq(labels, predictions).sum().float()
        f1_score = torch.div(true_positive, len(labels))
        return f1_score

    @staticmethod
    def calc_f1_count_for_label(predictions: torch.Tensor,
                                labels: torch.Tensor, label_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate f1 and true count for the label

        Args:
            predictions: tensor with predictions
            labels: tensor with original labels
            label_id: id of current label

        Returns:
            f1 score and true count for label
        """
        # label count
        true_count = torch.eq(labels, label_id).sum()

        # true positives: labels equal to prediction and to label_id
        true_positive = torch.logical_and(
            torch.eq(labels, predictions),
            torch.eq(labels, label_id)).sum().float()
        # precision for label
        precision = torch.div(true_positive, torch.eq(
            predictions, label_id).sum().float())
        # replace nan values with 0
        precision = torch.where(
            torch.isnan(precision),
            torch.zeros_like(precision).type_as(true_positive),
            precision
        )

        # recall for label
        recall = torch.div(true_positive, true_count)
        # f1
        f1 = 2 * precision * recall / (precision + recall)
        # replace nan values with 0
        f1 = torch.where(torch.isnan(f1), torch.zeros_like(
            f1).type_as(true_positive), f1)
        return f1, true_count

    def __call__(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate f1 score based on averaging method defined in init.

        Args:
            predictions: tensor with predictions
            labels: tensor with original labels

        Returns:
            f1 score
        """

        # simpler calculation for micro
        if self.average == 'micro':
            return self.calc_f1_micro(predictions, labels)

        f1_score = 0
        for label_id in range(1, len(labels.unique()) + 1):
            f1, true_count = self.calc_f1_count_for_label(
                predictions, labels, label_id)

            if self.average == 'weighted':
                f1_score += f1 * true_count
            elif self.average == 'macro':
                f1_score += f1

        if self.average == 'weighted':
            f1_score = torch.div(f1_score, len(labels))
        elif self.average == 'macro':
            f1_score = torch.div(f1_score, len(labels.unique()))

        return f1_score


def train_model(
    model: torch.nn.Module,
    criterion: torch.nn.modules.loss,
    optimizer: torch.optim,
    scheduler: torch.optim.lr_scheduler,
    dataloaders: dict,
    dataset_sizes: dict,
    device: torch.device,
    num_epochs: int = 25
) -> torch.nn.Module:
    """[Method of training pytorch model and returning best one based on accuracy]

    Args:
        model (torch.nn.Module): [Full pytorch model]
        criterion (torch.nn.modules.loss): [Loss criterion]
        optimizer (torch.optim): [Loss optimizer]
        scheduler (torch.optim.lr_scheduler): [Model learning rate scheduler]
        dataloaders (dict): [Dictionary of dataloaders for traning and validation]
        dataset_sizes (dict): [Dictionary of dataset sizes for traning and validation]
        device (str): [Cuda or CPU]
        num_epochs (int, optional): [Number of epochs]. Defaults to 25.

    Returns:
        torch.nn.Module: [Best model based on accuracy]
    """
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print('----------------')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                # move data to GPU
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # update learning rate with scheduler
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} loss: {epoch_loss:.4f} acc: {epoch_acc:.4f}")

            # deep copy the model with best accuracy on validation set
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(
        f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val acc: {best_acc:4f}")

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def test_model(
    model: torch.nn.Module,
    test_data: torchvision.datasets.folder.ImageFolder,
    len_data: int,
    device: torch.device,
    optimizer: torch.optim,
    criterion: torch.nn.modules.loss,
    f1: F1Score
) -> dict:
    """[Method for model evaluation based on Accuracy,F1 score and loss]

    Args:
        model (torch.nn.Module): [Full pytorch model]
        test_data (torchvision.datasets.folder.ImageFolder): [Test data from imagefolder]
        len_data (int): [Size of test data]
        device (str): [Cuda or CPU]
        optimizer (torch.optim): [Model optimizer]
        criterion (torch.nn.modules.loss): [Loss criterion]
        f1 (F1Score): [F1 score class]

    Returns:
        dict: [Dictionary with model evaluation]
    """
    running_loss = 0.0
    running_corrects = 0
    f1_metric = F1Score(f1)
    full_predictions = []
    full_labels = []

    for inputs, labels in test_data:
        model.eval()   # Set model to evaluate mode

        # move data to GPU
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            full_predictions.append(preds)
            full_labels.append(labels)
        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    full_f1 = f1_metric(torch.cat(full_predictions), torch.cat(full_labels))
    full_loss = running_loss / len_data
    full_accuracy = running_corrects / len_data
    full_accuracy = full_accuracy.item()
    final_dict = {
        "Accuracy": round(full_accuracy, 3),
        "F1 score": round(full_f1.item(), 3),
        "Loss": round(full_loss, 3)
    }
    return final_dict


def getAllPathsToFilesInDirectory(
    directory_name: str,
    accepted_filetypes: list = ['jpg', 'png', 'jpeg', 'JPG']
) -> pd.DataFrame:
    """ Get paths to all files in the directory 'directory_name' with accepted
    filetypes.

    This method extracts all files with accepted filetypes from
    'directory_name' and all of its subdirectories.

    Args:
        folder_name (str): Name of the folder we want to extract all the file
            paths from.

    Returns:
        pd.DataFrame: DataFrame with columns 'filename' and 'path_to_file'.
    """

    filename_list = []
    path_to_file_list = []

    for root, _, filenames in os.walk(directory_name):
        for filename in filenames:
            filetype = filename.split('.')[-1]
            if filetype in accepted_filetypes:
                filename_list.append(filename)
                path_to_file_list.append(os.path.join(root, filename))

    df = pd.DataFrame(
        data=list(zip(filename_list, path_to_file_list)),
        columns=['filename', 'path_to_file']
    )

    return df


def train_val_test_split(
    root_dir: str,
    train: float,
    val: float,
    del_data: bool = True,
    random_seed: int = None
) -> None:
    """This function inherit classes names based on names of folders with
    images in root folder then randomly shuffles and splits data into
    specified ratio of train-validation-test.

    Args:
        root_dir (str): Main Folder with other folders wich stored images
            separated by classes
        train (float): ratio of train set
        val (float): ratio of validation set
        test (float): ratio of test set
        del_data (bool): If True will delete initial folders with images
        random_seed (None) : Set seed to arbitrary int. number to get same
            split result.
    """
    np.random.seed(random_seed)
    assert train + val <= 1, "The ratios sum can not be more than 1"

    # Create new folders structure based on classes
    class_list = os.listdir(root_dir)

    for distinct_class in class_list:
        os.makedirs(root_dir + '/train/' + distinct_class)
        os.makedirs(root_dir + '/val/' + distinct_class)
        os.makedirs(root_dir + '/test/' + distinct_class)

    # Creating partitions of the data after shuffeling
    for distinct_class in class_list:
        source = root_dir + "/" + distinct_class  # Folder to copy images from

        allFileNames = os.listdir(source)
        np.random.shuffle(allFileNames)
        train_FileNames, val_FileNames, test_FileNames = np.split(
            np.array(allFileNames),
            [
                int(len(allFileNames) * train),
                int(len(allFileNames) * (train + val))
            ]
        )

        train_FileNames = [
            source + '/' + name for name in train_FileNames.tolist()
        ]
        val_FileNames = [
            source + '/' + name for name in val_FileNames.tolist()
        ]
        test_FileNames = [
            source + '/' + name for name in test_FileNames.tolist()
        ]

        print('Total images: ', "class:", distinct_class, len(allFileNames))
        print('Training: ', len(train_FileNames))
        print('Validation: ', len(val_FileNames))
        print('Testing: ', len(test_FileNames))

        # Copy-pasting images
        for name in train_FileNames:
            shutil.copy(name, root_dir + "/train/" + distinct_class)

        for name in val_FileNames:
            shutil.copy(name, root_dir + "/val/" + distinct_class)

        for name in test_FileNames:
            shutil.copy(name, root_dir + "/test/" + distinct_class)

    if del_data:
        for distinct_class in class_list:
            shutil.rmtree(root_dir + "/" + distinct_class, ignore_errors=True)
