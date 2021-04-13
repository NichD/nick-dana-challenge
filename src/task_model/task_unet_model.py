import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Activation, BatchNormalization
from tensorflow.keras.layers import UpSampling2D, Input, Concatenate, Reshape, Permute
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from task_model.attention import UNET_att_right


DEFAULT_IMAGE_SIZE = [512, 512, 3]
DEFAULT_MODEL_WIDTH_ALPHA = 0.5
DEFAULT_RESIDUAL_LAYER_NAMES = ['encoder_input',   
                                'block_1_expand_relu',   # 64x64x96
                                'block_3_expand_relu',   # 32x32x144
                                'block_6_expand_relu']   # 16x16x192
DEFAULT_RESIDUAL_LAYER_CHANNELS =  [16, 32, 48, 64]


def MobileNetV2_UNet(img_size: list=DEFAULT_IMAGE_SIZE, width_alpha: float=DEFAULT_MODEL_WIDTH_ALPHA,
                     residual_layer_names: list=DEFAULT_RESIDUAL_LAYER_NAMES,
                     residual_layer_channels: list=DEFAULT_RESIDUAL_LAYER_CHANNELS,
                     num_channels=1):
    # include norm layers - Norm weights taken from pytorch docs as TF docs are sparse
    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing/Resizing
    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing/Rescaling
    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing/Normalization
    norm_layers = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Resizing(img_size[0], img_size[1], name='resize'),
        tf.keras.layers.experimental.preprocessing.Rescaling(1./255, name='rescale'),
        tf.keras.layers.experimental.preprocessing.Normalization(axis=1,
                                                                 mean=[0.485, 0.456, 0.406],
                                                                 variance=[0.229**2, 0.224**2, 0.225**2],
                                                                 name='normalization')
                                                                 ])
    
    # include augmentation layer, it SHOULD apply only during .fit(...) 
    # https://www.tensorflow.org/guide/keras/transfer_learning
    # https://www.tensorflow.org/guide/keras/preprocessing_layers
    augmenation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomContrast(0.15, name='contrast'),
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal", name='rand_flip_horz'),
        tf.keras.layers.experimental.preprocessing.RandomFlip("vertical", name='rand_flip_vert'),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.05, name='rand_rot'), # +- 18 degree rotation
        tf.keras.layers.experimental.preprocessing.RandomZoom(0.05, name='rand_zoom'),  # +-5% zoom 
        tf.keras.layers.experimental.preprocessing.RandomTranslation(0.05, 0.05, name='rand_translate') # +-5% translation option
        ])    

    # get input apply norm and augment layers
    # defines model input single sample shape
    inputs = Input(shape=img_size, name="input_image")
    x = norm_layers(inputs)
    x = augmenation(x)
        
    # https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2
    # Let's breakdown this really quick:
    #     input_shape -> defines input sample size
    #     input_tensor -> optional arg also specifies input sample size
    #     weights -> indicates to use pretrained weights from imagenet
    #     include_top -> indicates whether to retain FCN from original
    #     alpha -> controls network "width", controls filter scaling down the network
    #     encoder_output -> top layer of encoder to start decode path
    # instantiate encoder, lock weights, get output.
    encoder = MobileNetV2(input_tensor=Input(name="encoder_input", tensor=x),
                          input_shape=img_size, weights="imagenet", 
                          include_top=False, alpha=width_alpha)
    encoder.trainable = False
    encoder_output = encoder.get_layer("block_13_expand_relu").output
    
    # Possible opportunity to use inpainting method. Here's some references that may be implementable
    # IMAGE INPAINTING FOR HIGH-RESOLUTION TEXTURES USING CNN TEXTURE SYNTHESIS https://arxiv.org/abs/1712.03111
    # https://towardsdatascience.com/how-copy-and-paste-is-embedded-in-cnns-for-image-inpainting-review-shift-net-image-433a2a93c963
    
    # encode features
    # _ = encoder(x, training=False) # just to be sure
    
    #  build decode path from the bottom-up
    x = encoder_output
    for i in range(1, len(residual_layer_names)+1, 1):
        # Upconv (scales input tensor up by 2x2) and concatenate residual layer (x_skip)
        x_skip = encoder.get_layer(residual_layer_names[-i]).output
        x = UpSampling2D((2, 2))(x)
        x = Concatenate()([x, x_skip])
        
        # apply 3x3 conv, preserve size (padding="same"), normalize and activate
        x = Conv2D(residual_layer_channels[-i], (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("LeakyReLU")(x)
        
        # apply 3x3 conv, preserve size (padding="same"), normalize and activate
        x = Conv2D(residual_layer_channels[-i], (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("LeakyReLU")(x)
    

    
    # https://github.com/keras-team/keras/issues/3653
    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Reshape
    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Permute
    # See above solution to add class-weights feature - AND THAT DIDN'T WORK!?!?!?!!!
    #   I'm going to implement weights in the loss-function directly - This is BS - Blargh!
    # x = Reshape(target_shape=(img_size[0]*img_size[1], img_size[2]), name='weights_reshape')(x)
    # x = Permute(dims=(2, 1))(x) # will need to reshape on output and verify Y is correct shape
    
    # top-level FCN (classes conv) and activation
    x = Conv2DTranspose(num_channels, (1, 1), padding="same")(x)
    outputs = Activation("softmax")(x)
    
    model = Model(inputs, outputs)
    return model


################ Alternative formulation to calculate image with mean RGB in the mask region
def MobileNetV2_UNet_Attn(img_size: list=DEFAULT_IMAGE_SIZE, width_alpha: float=DEFAULT_MODEL_WIDTH_ALPHA,
                     residual_layer_names: list=DEFAULT_RESIDUAL_LAYER_NAMES,
                     residual_layer_channels: list=DEFAULT_RESIDUAL_LAYER_CHANNELS,
                     num_channels=1):
    # include norm layers - Norm weights taken from pytorch docs as TF docs are sparse
    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing/Resizing
    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing/Rescaling
    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing/Normalization
    norm_layers = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Resizing(img_size[0], img_size[1], name='resize'),
        tf.keras.layers.experimental.preprocessing.Rescaling(1./255, name='rescale'),
        tf.keras.layers.experimental.preprocessing.Normalization(axis=1,
                                                                 mean=[0.485, 0.456, 0.406],
                                                                 variance=[0.229**2, 0.224**2, 0.225**2],
                                                                 name='normalization')
                                                                 ])
    
    # include augmentation layer, it SHOULD apply only during .fit(...) 
    # https://www.tensorflow.org/guide/keras/transfer_learning
    # https://www.tensorflow.org/guide/keras/preprocessing_layers
    augmenation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomContrast(0.5, name='contrast'),
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal", name='rand_flip_horz'),
        tf.keras.layers.experimental.preprocessing.RandomFlip("vertical", name='rand_flip_vert'),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.05, name='rand_rot'),
        tf.keras.layers.experimental.preprocessing.RandomZoom(0.1, name='rand_zoom'),
        tf.keras.layers.experimental.preprocessing.RandomTranslation(0.05, 0.05, name='rand_translate') 
        ])    

    # get input apply norm and augment layers
    # defines model input single sample shape
    inputs = Input(shape=img_size, name="input_image")
    x = norm_layers(inputs)
    x = augmenation(x)
    
    # https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2
    # Let's breakdown this really quick:
    #     input_shape -> defines input sample size
    #     input_tensor -> optional arg also specifies input sample size
    #     weights -> indicates to use pretrained weights from imagenet
    #     include_top -> indicates whether to retain FCN from original
    #     alpha -> controls network "width", controls filter scaling down the network
    #     encoder_output -> top layer of encoder to start decode path
    # instantiate encoder, lock weights, get output.
    encoder = MobileNetV2(input_tensor=Input(name="encoder_input", tensor=inputs),
                          input_shape=img_size, weights="imagenet", 
                          include_top=False, alpha=width_alpha)
    encoder.trainable = False
    encoder_output = encoder.get_layer("block_13_expand_relu").output
    
    # encode features
    # _ = encoder(x, training=False) # just to be sure
    
    #  build decode path from the bottom-up
    # Modeled after https://github.com/yingkaisha/keras-unet-collection/blob/main/keras_unet_collection/_model_att_unet_2d.py
    # implementation of https://www.sciencedirect.com/science/article/pii/S1361841518306133
    x = encoder_output
    for i in range(1, len(residual_layer_names)+1, 1):
        # combined upconv, attention and concat blocks
        filt_ch = residual_layer_channels[-i]
        x_skip = encoder.get_layer(residual_layer_names[-i]).output
        x = UNET_att_right(x, x_skip, filt_ch, filt_ch//2, batch_norm=True,
                           name=f'attn_up{i}', activation='LeakyReLU')
    
    # top-level (num channels) layer for MSE loss propagation
    x = Conv2DTranspose(num_channels, (1, 1), padding="same")(x)
    outputs = Activation("sigmoid")(x)
    
    model = Model(inputs, outputs)
    return model




################ Alternative formulation to calculate image with mean RGB in the mask region
def MobileNetV2_UNet_MeanMask(img_size: list=DEFAULT_IMAGE_SIZE, width_alpha: float=DEFAULT_MODEL_WIDTH_ALPHA,
                     residual_layer_names: list=DEFAULT_RESIDUAL_LAYER_NAMES,
                     residual_layer_channels: list=DEFAULT_RESIDUAL_LAYER_CHANNELS,
                     num_channels=1):
    # include norm layers - Norm weights taken from pytorch docs as TF docs are sparse
    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing/Resizing
    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing/Rescaling
    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing/Normalization
    norm_layers = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Resizing(img_size[0], img_size[1], name='resize'),
        tf.keras.layers.experimental.preprocessing.Rescaling(1./255, name='rescale'),
        tf.keras.layers.experimental.preprocessing.Normalization(axis=1,
                                                                 mean=[0.485, 0.456, 0.406],
                                                                 variance=[0.229**2, 0.224**2, 0.225**2],
                                                                 name='normalization')
                                                                 ])
    
    # include augmentation layer, it SHOULD apply only during .fit(...) 
    # https://www.tensorflow.org/guide/keras/transfer_learning
    # https://www.tensorflow.org/guide/keras/preprocessing_layers
    augmenation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomContrast(0.15, name='contrast'),
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal", name='rand_flip_horz'),
        tf.keras.layers.experimental.preprocessing.RandomFlip("vertical", name='rand_flip_vert'),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.05, name='rand_rot'), # +- 18 degree rotation
        tf.keras.layers.experimental.preprocessing.RandomZoom(0.05, name='rand_zoom'),  # +-5% zoom 
        tf.keras.layers.experimental.preprocessing.RandomTranslation(0.05, 0.05, name='rand_translate') # +-5% translation option
        ])    

    # get input apply norm and augment layers
    # defines model input single sample shape
    inputs = Input(shape=img_size, name="input_image")
    x = norm_layers(inputs)
    x = augmenation(x)
        
    # https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2
    # Let's breakdown this really quick:
    #     input_shape -> defines input sample size
    #     input_tensor -> optional arg also specifies input sample size
    #     weights -> indicates to use pretrained weights from imagenet
    #     include_top -> indicates whether to retain FCN from original
    #     alpha -> controls network "width", controls filter scaling down the network
    #     encoder_output -> top layer of encoder to start decode path
    # instantiate encoder, lock weights, get output.
    encoder = MobileNetV2(input_tensor=Input(name="encoder_input", tensor=inputs),
                          input_shape=img_size, weights="imagenet", 
                          include_top=False, alpha=width_alpha)
    encoder.trainable = False
    encoder_output = encoder.get_layer("block_13_expand_relu").output
    
    # encode features
    # _ = encoder(x, training=False) # just to be sure
    
    #  build decode path from the bottom-up
    x = encoder_output
    for i in range(1, len(residual_layer_names)+1, 1):
        # Upconv (scales input tensor up by 2x2) and concatenate residual layer (x_skip)
        x_skip = encoder.get_layer(residual_layer_names[-i]).output
        x = UpSampling2D((2, 2))(x)
        x = Concatenate()([x, x_skip])
        
        # apply 3x3 conv, preserve size (padding="same"), normalize and activate
        x = Conv2D(residual_layer_channels[-i], (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("")(x)
        
        # apply 3x3 conv, preserve size (padding="same"), normalize and activate
        x = Conv2D(residual_layer_channels[-i], (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
    
    # top-level (num channels) layer for MSE loss propagation
    x = Conv2DTranspose(num_channels, (1, 1), padding="same")(x)
    outputs = Activation("relu")(x)
    
    model = Model(inputs, outputs)
    return model
