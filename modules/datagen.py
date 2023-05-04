import os
from enum import Enum
import cv2
import numpy as np
from PIL import Image
from keras.utils import to_categorical
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_masks(directory_path, patch_size):
    """
    :param patch_size: image patchify square size
    :param directory_path: path to root directory containing training and test images
    :return: list of images from directory
    """

    # initialize empty list for images
    instances = []

    # iterate through files in directory
    for filepath in tqdm(sorted(os.listdir(directory_path))):
        extension = filepath.split(".")[-1]
        if extension == "jpg" or extension == "png":

            # current image path
            img_path = rf"{directory_path}/{filepath}"
            #print(img_path)

            # convert image to RBG
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            thresh = 127
            max_val = 255
            

            size_x = (image.shape[1] // patch_size) * patch_size  # width to the nearest size divisible by patch size
            size_y = (image.shape[0] // patch_size) * patch_size  # height to the nearest size divisible by patch size

            image = Image.fromarray(image)

            # Crop original image to size divisible by patch size from top left corner
            image = np.array(image.crop((0, 0, size_x, size_y)))
            image = cv2.resize(image, (854, 854))
            image = cv2.resize(image, (160, 160))

            image = cv2.threshold(image, thresh, max_val, cv2.THRESH_BINARY)[1]
            image = image/255
            #plt.imshow(image)
            #plt.show()
            instances.append(np.squeeze(image))

    return instances

def load_images(directory_path, patch_size):
    """
    :param patch_size: image patchify square size
    :param directory_path: path to root directory containing training and test images
    :return: list of images from directory
    """

    # initialize empty list for images
    instances = []

    # iterate through files in directory
    for filepath in tqdm(sorted(os.listdir(directory_path))):
        extension = filepath.split(".")[-1]
        if extension == "jpg" or extension == "png":

            # current image path
            img_path = rf"{directory_path}/{filepath}"
            #print(img_path)

            # Reads image as BGR
            image = cv2.imread(img_path)

            # convert image to RBG
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            size_x = (image.shape[1] // patch_size) * patch_size  # width to the nearest size divisible by patch size
            size_y = (image.shape[0] // patch_size) * patch_size  # height to the nearest size divisible by patch size

            image = Image.fromarray(image)

            # Crop original image to size divisible by patch size from top left corner
            image = np.array(image.crop((0, 0, size_x, size_y)))
            image = cv2.resize(image, (854, 854))
            image = cv2.resize(image, (160, 160))
            instances.append(np.squeeze(image))

    return instances

def display_images(instances, rows=2, titles=None):
    """
    :param instances:  list of images
    :param rows: number of rows in subplot
    :param titles: subplot titles
    :return:
    """
    n = len(instances)
    cols = n // rows if (n / rows) % rows == 0 else (n // rows) + 1
    
    # iterate through images and display subplots
    for j, image in enumerate(instances):
        plt.subplot(rows, cols, j + 1)
        plt.title('') if titles is None else plt.title(titles[j])
        plt.axis("off")
        #print("hello")
        plt.imshow(image)
        cnt = random.uniform(0,100)
        cv2.imwrite(os.getcwd()+'/outputs/'+str(cnt)+'.jpg', image)
        #print("written to this path: ",os.getcwd()+'/outputs/'+str(cnt)+'.jpg' )
    # show the figure
    plt.show()

# =====================================================
# prepare training data input images

def get_training_data(root_directory):
    # initialise lists
    image_dataset, mask_dataset = [], []
    ip, mp = [], []

    # define image patch size
    patch_size = 160
    
    # walk through root directory
    for path, directories, files in os.walk(root_directory):
        for subdirectory in directories:
            #print("subdirect",subdirectory)
            # extract training input images and patchify
            if subdirectory == "images":
                image_dataset.extend(
                    load_images(os.path.join(path, subdirectory), patch_size=patch_size))

            # extract training label masks and patchify
            elif subdirectory == "masks":
                mask_dataset.extend(
                    load_masks(os.path.join(path, subdirectory), patch_size=patch_size))
    # return input images and masks
    return np.array(image_dataset), np.array(mask_dataset), 


def create_binary_segmentation_problem(image_dataset, mask_dataset):
    # change problem to binary segmentation problem
    x_reduced, y_reduced = [], []

    # iterate over masks
    for j, mask in tqdm(enumerate(mask_dataset)):

        # get image shape
        _img_height, _img_width, _img_channels = mask.shape

        # create binary image (zeros)
        binary_image = np.zeros((_img_height, _img_width, 1)).astype(int)

        # iterate over each pixel in mask
        for row in range(_img_height):
            for col in range(_img_width):
                # get image channel across axis=3
                rgb = mask[row, col, :]

                # building hex: #3C1098 = RGB(60, 16, 152) or BGR(152, 16, 60)
                binary_image[row, col] = 1 if rgb[0] == 60 and rgb[1] == 16 and rgb[2] == 152 else 0

        # only keep images with a high percentage of building coverage
        if np.count_nonzero(binary_image == 1) > 0.15 * binary_image.size:
            x_reduced.append(image_dataset[j])
            y_reduced.append(binary_image)

    # return binary image dataset
    return np.array(x_reduced), np.array(y_reduced)


# mask color codes
class MaskColorMap(Enum):
    car = (255, 255, 255)
    no_car = (0, 0, 0)

def one_hot_encode_masks(masks, num_classes):
    """
    :param masks: Y_train patched mask dataset
    :param num_classes: number of classes
    :return:
    """
    # initialise list for integer encoded masks
    """mapper = {x:[0]*num_classes for x in range(num_classes)}
    for k in mapper:
      temp = mapper[k]
      temp[k] = 1
      mapper[k] = temp"""

    integer_encoded_labels = []

    # iterate over each mask
    for mask in tqdm(masks):

        # get image shape
        _img_height, _img_width, _img_channel = mask.shape

        # create new mask of zeros
        encoded_image = np.zeros((_img_height, _img_width, 1)).astype(int)

        for j, cls in enumerate(MaskColorMap):
            encoded_image[np.all(mask == cls.value, axis=-1)] = j

        # append encoded image
        integer_encoded_labels.append(encoded_image)

    # return one-hot encoded labels
    return to_categorical(y=integer_encoded_labels, num_classes=num_classes)

def get_data(data_dir):
    n_classes = 2
    train_size = -1

    # create (X, Y) training data
    X, Y = get_training_data(root_directory=data_dir)

    X = X[:train_size]
    Y = Y[:train_size]
    Y = np.expand_dims(Y, axis=-1)
        
    # extract X_train shape parameters
    m, img_height, img_width, img_channels = X.shape

    # display images from both training and test sets
    display_count = 6
    random_index = [np.random.randint(0, m) for _ in range(display_count)]

    # convert RGB values to integer encoded labels for categorical_crossentropy
    Y = one_hot_encode_masks(Y, num_classes=n_classes)
    X = X/255 # rescaling

    # split dataset into training and test groups
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)

    return X_train, X_test, Y_train, Y_test