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

def UNet_3Plus(INPUT_SHAPE, OUTPUT_CHANNELS):
    filters = [64, 128, 256, 512, 1024]

    input_layer = k.layers.Input(shape=INPUT_SHAPE, name="input_layer")  # (2,512,512,3)

    """ Encoder"""
    # block 1
    e1 = conv_block(input_layer, filters[0])  # (2,512,512,64)

    # block 2
    e2 = k.layers.MaxPool2D(pool_size=(2, 2))(e1)  # (2,256,256,64)
    e2 = conv_block(e2, filters[1])  # (2,256,256,128)

    # block 3
    e3 = k.layers.MaxPool2D(pool_size=(2, 2))(e2)  # (2,128,128,128)
    e3 = conv_block(e3, filters[2])  # (2,128,128,256)

    # block 4
    e4 = k.layers.MaxPool2D(pool_size=(2, 2))(e3)  # (2,64,64,256)
    e4 = conv_block(e4, filters[3])  #  (2,64,64,512)

    # block 5
    # bottleneck layer
    e5 = k.layers.MaxPool2D(pool_size=(2, 2))(e4)  # (2,32,32,512)
    e5 = conv_block(e5, filters[4])  # (2,32,32,1024)

    """ Decoder """

    """ d4 """
    e1_d4 = k.layers.MaxPool2D(pool_size=(8, 8))(e1)  # (2,64,64,64)
    e1_d4 = conv_block(e1_d4, filters[0], n=1)  # (2,64,64,64)

    e2_d4 = k.layers.MaxPool2D(pool_size=(4, 4))(e2)  # (2,64,64,128)
    e2_d4 = conv_block(e2_d4, filters[1], n=1)   # (2,64,64,128)

    e3_d4 = k.layers.MaxPool2D(pool_size=(2, 2))(e3)  # (2,64,64,256)
    e3_d4 = conv_block(e3_d4, filters[2], n=1)  # (2,64,64,256)

    e4_d4 = conv_block(e4, filters[3], n=1)  # (2,64,64,512)

    e5_d4 = k.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(e5) # (2,64,64,512)
    e5_d4 = conv_block(e5_d4, filters[3], n=1)  # (2,64,64,512)

    d4 = k.layers.concatenate([e1_d4, e2_d4, e3_d4, e4_d4, e5_d4])
    d4 = conv_block(d4, filters[3], n=1)  # (2,64,64,512)

    """ d3 """
    e1_d3 = k.layers.MaxPool2D(pool_size=(4, 4))(e1)
    e1_d3 = conv_block(e1_d3, filters[0], n=1)

    e2_d3 = k.layers.MaxPool2D(pool_size=(2, 2))(e2)
    e2_d3 = conv_block(e2_d3, filters[1], n=1)

    e3_d3 = conv_block(e3, filters[2], n=1)

    e4_d3 = k.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(d4)
    e4_d3 = conv_block(e4_d3, filters[2], n=1)

    e5_d3 = k.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(e5)
    e5_d3 = conv_block(e5_d3, filters[2], n=1)

    d3 = k.layers.concatenate([e1_d3, e2_d3, e3_d3, e4_d3, e5_d3])
    d3 = conv_block(d3, filters[2], n=1)

    """ d2 """
    e1_d2 = k.layers.MaxPool2D(pool_size=(2, 2))(e1)
    e1_d2 = conv_block(e1_d2, filters[0], n=1)

    e2_d2 = conv_block(e2, filters[1], n=1)

    d3_d2 = k.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(d3)
    d3_d2 = conv_block(d3_d2, filters[1], n=1)

    d4_d2 = k.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(d4)
    d4_d2 = conv_block(d4_d2, filters[1], n=1)

    e5_d2 = k.layers.UpSampling2D(size=(8, 8), interpolation='bilinear')(e5)
    e5_d2 = conv_block(e5_d2, filters[1], n=1)

    d2 = k.layers.concatenate([e1_d2, e2_d2, d3_d2, d4_d2, e5_d2])
    d2 = conv_block(d2, filters[1], n=1)

    """ d1 """
    e1_d1 = conv_block(e1, filters[0], n=1)

    d2_d1 = k.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(d2)
    d2_d1 = conv_block(d2_d1, filters[0], n=1)

    d3_d1 = k.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(d3)
    d3_d1 = conv_block(d3_d1, filters[0], n=1)

    d4_d1 = k.layers.UpSampling2D(size=(8, 8), interpolation='bilinear')(d4)
    d4_d1 = conv_block(d4_d1, filters[0], n=1)

    e5_d1 = k.layers.UpSampling2D(size=(16, 16), interpolation='bilinear')(e5)
    e5_d1 = conv_block(e5_d1, filters[0], n=1)

    d1 = k.layers.concatenate([e1_d1, d2_d1, d3_d1, d4_d1, e5_d1, ])
    d1 = conv_block(d1, filters[0], n=1)

    # last layer does not have batchnorm and relu
    d = conv_block(d1, OUTPUT_CHANNELS, n=1, is_bn=False, is_relu=False)

    if OUTPUT_CHANNELS == 1:
        output = k.activations.sigmoid(d)
    else:
        output = k.activations.softmax(d)

    return tf.keras.Model(inputs=input_layer, outputs=output, name='UNet_3Plus')

# if __name__ == "__main__":
#     input_shape = (512, 512, 3)
#     OUTPUT_CHANNELS = 1
#     model = UNet_3Plus(input_shape, OUTPUT_CHANNELS)
#     metrics = ['accuracy',recall_m,jacard_coef_loss]
#     model.compile(k.optimizers.Adam(learning_rate=0.0001), loss=dice_loss, metrics=metrics)

#     model.summary()