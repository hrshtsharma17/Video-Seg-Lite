import os
import hls4ml
from keras import backend as K
from keras.models import load_model

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

def gen_hls4ml_model(model, out_dir, fpga_xylinx):
    hls_config = hls4ml.utils.config_from_keras_model(model, granularity='name')

    # Set the precision and reuse factor for the full model
    hls_config['Model']['Precision'] = 'ap_fixed<16,6>'
    hls_config['Model']['ReuseFactor'] = 1

    # Create an entry for each layer, here you can for instance change the strategy for a layer to 'resource' 
    # or increase the reuse factor individually for large layers.
    # In this case, we designed the model to be small enough for a fully parallel implementation 
    # so we use the latency strategy and reuse factor of 1 for all layers.
    for Layer in hls_config['LayerName'].keys():
        hls_config['LayerName'][Layer]['Strategy'] = 'latency'
        hls_config['LayerName'][Layer]['ReuseFactor'] = 1

    #If you want best numerical performance for high-accuray models, while the default latency strategy is faster but numerically more unstable
    sm_layer = list(hls_config["LayerName"].keys())[-1]
    hls_config['LayerName'][sm_layer]['Strategy'] = 'stable'

    cfg = hls4ml.converters.create_config(backend='Vivado')
    cfg['IOType']     = 'io_stream' # Must set this if using CNNs!
    cfg['HLSConfig']  = hls_config
    cfg['KerasModel'] = model
    cfg['OutputDir']  = out_dir
    cfg['XilinxPart'] = fpga_xylinx
    #cfg['XilinxPart'] = "xczu9eg-2ffvb1156"

    
    hls_model = hls4ml.converters.keras_to_hls(cfg)
    return hls_model

if __name__=="__main__":
    model_dir = os.getcwd()+'/models/'
    model_name = 'cv_unet_lite_davis_450k-12k_distill_2023-05-01 14_52_03_553482.hdf5'

    model = load_model(
        model_dir + model_name,
        custom_objects={'iou_coefficient': iou_coefficient, 'jaccard_index': jaccard_index}
    )

    hls_model = gen_hls4ml_model(model, 'fpga_model_400to12k_xcu250/', 'xcu250-figd2104-2L-e')

    # compile, plot and eval
    hls_model.compile()
    hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file="UNL_12distill.png")

