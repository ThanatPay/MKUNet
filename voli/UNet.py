import tensorflow as tf
import tensorflow.keras as k
from tensorflow.keras.metrics import Recall, Precision

def conv_block(x, kernels, kernel_size=(3, 3), strides=(1, 1), padding='same',
               is_bn=True, is_relu=True, n=2):
    """ Custom function for conv2d:
        Apply  3*3 convolutions with BN and relu.
    """
    for i in range(1, n + 1):
        x = k.layers.Conv2D(filters=kernels, kernel_size=kernel_size,
                            padding=padding, strides=strides,
                            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                            kernel_initializer=k.initializers.he_normal(seed=5))(x)
        if is_bn:
            x = k.layers.BatchNormalization()(x)
        if is_relu:
            x = k.activations.relu(x)

    return x

def dotProduct(seg, cls):
    B, H, W, N = k.backend.int_shape(seg)
    seg = tf.reshape(seg, [-1, H * W, N])
    final = tf.einsum("ijk,ik->ijk", seg, cls)
    final = tf.reshape(final, [-1, H, W, N])
    return final

def UNet(INPUT_SHAPE, OUTPUT_CHANNELS):
    filters = [64, 128, 256, 512, 1024]

    input_layer = k.layers.Input(shape=INPUT_SHAPE, batch_size=2, name="input_layer")  # (2,512,512,3)

    """ Encoder"""
    # block 1
    e1 = conv_block(input_layer, filters[0])  # (2,512,512,64)

    # block 2
    e2 = k.layers.MaxPool2D(pool_size=(2, 2))(e1)  # (2,256,256,64)
    e2 = conv_block(e2, filters[1])  # (2,256,256,128)
    # e2 = k.layers.Dropout(0.1)(e2)

    # block 3
    e3 = k.layers.MaxPool2D(pool_size=(2, 2))(e2)  # (2,128,128,128)
    e3 = conv_block(e3, filters[2])  # (2,128,128,256)
    # e3 = k.layers.Dropout(0.1)(e3)

    # block 4
    e4 = k.layers.MaxPool2D(pool_size=(2, 2))(e3)  # (2,64,64,256)
    e4 = conv_block(e4, filters[3])  # (2,64,64,512)
    e4 = k.layers.Dropout(0.1)(e4)

    #connection between encoder, decoder
    e5 = k.layers.MaxPool2D(pool_size=(2, 2))(e4)
    e5 = conv_block(e5, filters[4])  # (2,32,32,1024)
    e5 = k.layers.Dropout(0.1)(e5)

    """ Decoder """
    # decoder block 4
    d4 = k.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(e5) # (2,64,64,512)
    d4 = conv_block(d4, filters[3], n=1) # (2,64,64,512)
    d4 = k.layers.concatenate([d4, e4], axis=-1) # (2,64,64,1024)
    d4 = conv_block(d4, filters[3], n=1) # (2,64,64,512)

    # decoder block 3
    d3 = k.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(d4) # (2,128,128,512)
    d3 = conv_block(d3, filters[2], n=1) # (2,128,128,256)
    d3 = k.layers.concatenate([d3, e3], axis=-1) # (2,128,128,512)
    d3 = conv_block(d3, filters[2], n=1) # (2,128,128,256)

    # decoder block 2
    d2 = k.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(d3) # (2,256,256,256)
    d2 = conv_block(d2, filters[1], n=1) # (2,256,256,128)
    d2 = k.layers.concatenate([d2, e2], axis=-1) # (2,256,256,256)
    d2 = conv_block(d2, filters[1], n=1) # (2,256,256,128)

    # decoder block 1
    d1 = k.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(d2) # (2,512,512,128)
    d1 = conv_block(d1, filters[0], n=1) # (2,512,512,64)
    d1 = k.layers.concatenate([d1, e1], axis=-1) # (2,512,512,128)
    d1 = conv_block(d1, filters[0]) # (2,512,512,64)

    # last layer does not have batchnorm and relu
    d = conv_block(d1, OUTPUT_CHANNELS, n=1, is_bn=False, is_relu=False) # (2,512,512,1)

    if OUTPUT_CHANNELS == 1:
        output = k.activations.sigmoid(d)
    else:
        output = k.activations.softmax(d)

    return tf.keras.Model(inputs=input_layer, outputs=output, name='UNet')

# if __name__ == "__main__":
#     input_shape = (512, 512, 3)
#     OUTPUT_CHANNELS = 1
#     model = UNet(input_shape, OUTPUT_CHANNELS)
#     metrics = ['accuracy',recall_m,jacard_coef_loss]
#     model.compile(k.optimizers.Adam(learning_rate=0.00001), loss=dice_loss, metrics=metrics)

#     model.summary()