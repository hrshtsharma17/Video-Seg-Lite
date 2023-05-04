import os
import numpy as np
import datetime
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from modules.models import build_unet_v3
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

if __name__=="__main__":
    dt_now = str(datetime.datetime.now()).replace(".", "_").replace(":", "_")
    model_img_save_path = f"{os.getcwd()}/models/cv_enet_lite_davis_450k_{dt_now}.png"
    model_save_path = f"{os.getcwd()}/models/cv_enet_lite_davis_450k_{dt_now}.hdf5"
    model_checkpoint_filepath = os.getcwd() + "/models/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
    csv_logger = rf"{os.getcwd()}/logs/aerial_segmentation_log_{dt_now}.csv"

    X_train, X_test, Y_train, Y_test = get_data('./')

    model = build_unet_v3((160, 160, 3))
    print(model.summary())

    #checking if implementation is viable for VIVADO
    for layer in model.layers:
        if layer.__class__.__name__ in ['Conv2D', 'Dense']:
            w = layer.get_weights()[0]
            layersize = np.prod(w.shape)
            print("{}: {}".format(layer.name,layersize)) # 0 = weights, 1 = biases
            if (layersize > 4096): # assuming that shape[0] is batch, i.e., 'None'
                print("Layer {} is too large ({}), are you sure you want to train?".format(layer.name,layersize))
    
    # =======================================================
    # add callbacks, compile model and fit training data

    # save best model with maximum validation accuracy
    checkpoint = ModelCheckpoint(model_checkpoint_filepath, monitor="val_accuracy", verbose=1, save_best_only=True,
                                mode="max")

    # stop model training early if validation loss doesn't continue to decrease over 2 iterations
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, verbose=1, mode="min")

    # log training console output to csv
    csv_logger = CSVLogger(csv_logger, separator=",", append=False)

    # create list of callbacks
    callbacks_list = [early_stopping, csv_logger]  # early_stopping

    # compile model
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy", iou_coefficient, jaccard_index])

    # train and save model
    batch_size = 32
    model.fit(X_train, Y_train, epochs=50, batch_size=batch_size, validation_data=(X_test, Y_test), verbose=1, callbacks = callbacks_list)
        
    model.save(model_save_path)
    print("model saved:", model_save_path)

