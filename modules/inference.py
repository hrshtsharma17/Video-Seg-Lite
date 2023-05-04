import os
import numpy as np
from keras.models import load_model
from keras import backend as K
from datagen import display_images
from modules.datagen import get_data

# Mean Intersection-Over-Union: iou = true_positives / (true_positives + false_positives + false_negatives)
def iou_coefficient(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou


# jaccard similarity: the size of the intersection divided by the size of the union of two sets
def jaccard_index(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


def rgb_encode_mask(mask):
    # initialize rgb image with equal spatial resolution
    rgb_encode_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    # convert binary mask to RGB format
    rgb_encode_image[mask == 1] = (255, 255, 255)  # white for foreground
    rgb_encode_image[mask == 0] = (0, 0, 0)  # black for background

    return rgb_encode_image

if __name__=="__main__":
    model_dir = os.getcwd()+'/models/'
    model_name = 'cv_unet_lite_davis_450k-12k_distill_2023-05-01 14_52_03_553482.hdf5'

    model = load_model(
        model_dir + model_name,
        custom_objects={'iou_coefficient': iou_coefficient, 'jaccard_index': jaccard_index}
    )

    _, X_test, _, Y_test = get_data('./')

    for _ in range(10):
        # choose random number from 0 to test set size
        test_img_number = np.random.randint(0, len(X_test))

        # extract test input image
        test_img = X_test[test_img_number]

        # ground truth test label converted from one-hot to integer encoding
        ground_truth = np.argmax(Y_test[test_img_number], axis=-1)

        # expand first dimension as U-Net requires (m, h, w, nc) input shape
        test_img_input = np.expand_dims(test_img, 0)

        # make prediction with model and remove extra dimension
        prediction = np.squeeze(model.predict(test_img_input))
        
        # convert softmax probabilities to integer values
        predicted_img = np.argmax(prediction, axis=-1)

        #plt.imshow(predicted_img)
        # convert integer encoding to rgb values
        rgb_image = rgb_encode_mask(predicted_img)
        rgb_ground_truth = rgb_encode_mask(ground_truth)

        # visualize model predictions
        test_img = test_img
        display_images(
            [test_img, ground_truth, predicted_img],
            rows=1, titles=['Input Image', 'Ground Truth', 'Prediction']
        )

    # Test Scores
    test_score_baseline = model.evaluate(X_test, Y_test)

    print('Scores: ', test_score_baseline)