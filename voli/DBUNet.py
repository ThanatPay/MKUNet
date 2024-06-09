import tensorflow as tf
import tensorflow.keras as k
from einops.layers.tensorflow import Rearrange
from einops import repeat
from tensorflow.keras.metrics import Recall, Precision

def conv_block(x, channels, kernel_size=(3, 3), strides=(1, 1), padding='same',
               is_bn=True, is_relu=True, n=2):
    """ Custom function for conv2d:
        Apply  3*3 convolutions with BN and relu.
    """
    for i in range(1, n + 1):
        x = k.layers.Conv2D(filters=channels, kernel_size=kernel_size,
                            padding=padding, strides=strides,
                            kernel_regularizer=k.regularizers.l2(1e-4),
                            kernel_initializer=k.initializers.he_normal(seed=5))(x)
        if is_bn:
            x = k.layers.BatchNormalization()(x)
        if is_relu:
            x = k.activations.relu(x)

    return x

def convT_block(x, channels, kernel_size=(3, 3), strides=(1, 1), padding='same'):

    x = k.layers.Conv2DTranspose(filters=channels, kernel_size=kernel_size,
                                  strides=2, padding=padding,
                                  kernel_initializer = 'he_normal')(x)
    x = k.layers.BatchNormalization()(x)
    x = k.activations.relu(x)

    x = k.layers.Conv2D(filters=channels, kernel_size=(3,3),
                        padding=padding, strides=strides,
                        kernel_regularizer=k.regularizers.l2(1e-4),
                        kernel_initializer=k.initializers.he_normal(seed=5))(x)
    x = k.layers.BatchNormalization()(x)
    x = k.activations.relu(x)

    return x

def Bottleneck(x, channels):
    x = k.layers.Conv2D(filters=channels, kernel_size=(1,1), padding='same',
                        strides=(1,1), groups=4, use_bias=False,
                        kernel_regularizer=k.regularizers.l2(1e-4),
                        kernel_initializer=k.initializers.he_normal(seed=5))(x)
    x = k.layers.BatchNormalization()(x)

    residual1 = x

    x = k.layers.Conv2D(filters=channels, kernel_size=(3,3), padding='same',
                        strides=(1,1), groups=2, use_bias=False,
                        kernel_regularizer=k.regularizers.l2(1e-4),
                        kernel_initializer=k.initializers.he_normal(seed=5))(x)
    x = k.layers.BatchNormalization()(x)
    x = k.activations.relu(x)

    residual2 = x

    x = k.layers.Conv2D(filters=channels, kernel_size=(1,1), padding='same',
                        strides=(1,1), groups=4, use_bias=False,
                        kernel_regularizer=k.regularizers.l2(1e-4),
                        kernel_initializer=k.initializers.he_normal(seed=5))(x)
    x = k.layers.BatchNormalization()(x)
    x = k.activations.relu(x)

    x = x + residual1

    x = k.layers.Conv2D(filters=channels, kernel_size=(3,3), padding='same',
                        strides=(1,1), groups=2, use_bias=False,
                        kernel_regularizer=k.regularizers.l2(1e-4),
                        kernel_initializer=k.initializers.he_normal(seed=5))(x)
    x = k.layers.BatchNormalization()(x)
    x = k.activations.relu(x)

    x = x + residual2

    x = k.layers.Conv2D(filters=channels, kernel_size=(1,1), padding='same',
                        strides=(1,1), groups=4, use_bias=False,
                        kernel_regularizer=k.regularizers.l2(1e-4),
                        kernel_initializer=k.initializers.he_normal(seed=5))(x)
    x = k.layers.BatchNormalization()(x)
    x = k.activations.relu(x)
    return x

def dotProduct(seg, cls):
    B, H, W, N = k.backend.int_shape(seg)
    seg = tf.reshape(seg, [-1, H * W, N])
    final = tf.einsum("ijk,ik->ijk", seg, cls)
    final = tf.reshape(final, [-1, H, W, N])
    return final

def gelu(x, approximate=False):
    if approximate:
        coeff = tf.cast(0.044715, x.dtype)
        return 0.5 * x * (1.0 + tf.tanh(0.7978845608028654 * (x + coeff * tf.pow(x, 3))))
    else:
        return 0.5 * x * (1.0 + tf.math.erf(x / tf.cast(1.4142135623730951, x.dtype)))

def Attention(x, emb_dim, heads=8, dim_head=64, dropout=0.):
    inner_dim = dim_head * heads
    scale = dim_head**(-0.5)
    qkv = k.layers.Dense(inner_dim * 3)(x) # (2,257,3072)
    qkv = tf.split(qkv, 3, axis=-1)
    query, key, value = map(lambda t: Rearrange('b n (h d) -> b h n d', h=heads)(t), qkv)
    # print(query.shape)

    key_t = tf.transpose(key, perm=[0,1,3,2])
    # print(key_t.shape)
    dots = tf.matmul(query, key_t) * scale
    # print(dots.shape)

    attn = tf.nn.softmax(dots, axis=-1)
    attn = k.layers.Dropout(dropout)(attn)

    out = tf.matmul(attn, value)
    # print(out.shape)
    out = Rearrange('b h n d -> b n (h d)')(out)
    out = k.layers.Dense(emb_dim)(out)
    out = k.layers.Dropout(dropout)(out)

    # print(out.shape)
    return out

def Transformer(x, token, x_AfterPatchEmbedding, num_batch, num_patch, emb_dim, depth, heads, dim_head, mlp_dim, dropout=0.):
    x_AfterConv = tf.reshape(x_AfterPatchEmbedding, (num_batch*num_patch, int(emb_dim**0.5), int(emb_dim**0.5), 1)) # (128,32,32,1)
    x_AfterConv = k.layers.Conv2D(filters=1, kernel_size=3, padding= 'same', strides=1, use_bias=False)(x_AfterConv) # (128,32,32,1)
    x_AfterConv = k.layers.BatchNormalization()(x_AfterConv)
    x_AfterConv = k.activations.relu(x_AfterConv)

    a = tf.reshape(x_AfterConv, (num_batch, num_patch, 1, emb_dim))
    # print(a.shape) # (2,64,1,1024)
    b = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_first')(a)
    # print(b.shape) # (2,64)
    val = tf.sort(b, axis=1)
    s = tf.cast(val < 0, tf.float32)
    e = tf.cast(val >= 0, tf.float32)
    l_s = tf.reduce_sum(s)
    l_e = tf.reduce_sum(s)
    suppress = (val - (1 / (1 + tf.pow(l_s, val))))*s
    excitation = (val + (1 / (1 + tf.pow(l_e, val))))*e
    b_updated = suppress + excitation
    #suppress -> Inverse sigmoid // excitation -> sigmoid

    c = k.layers.Dense(num_patch, use_bias=False)(b_updated)
    c = k.activations.relu(c)
    c = k.layers.Dense(num_patch, use_bias=False)(c)
    c = k.activations.sigmoid(c)
    c = tf.reshape(c, (num_batch,num_patch,1,1)) # (2,64,1,1)


    x_attention = a * tf.repeat(c, emb_dim, axis=-1)
    x_attention = tf.reshape(x_attention, (num_batch,num_patch,emb_dim))
    x_attention = k.layers.concatenate([x_attention, token], axis=1) # (2,65,1024)

    for _ in range(depth):

        attn_x = k.layers.LayerNormalization()(x)
        attn_x = Attention(x, emb_dim, heads=heads, dim_head=dim_head, dropout=dropout)
        x = attn_x + x + x_attention

        ff_x = k.layers.LayerNormalization()(x)
        ff_x = k.layers.Dense(mlp_dim)(ff_x)
        ff_x = gelu(ff_x)
        ff_x = k.layers.Dropout(dropout)(ff_x)
        ff_x = k.layers.Dense(emb_dim)(ff_x)
        ff_x = k.layers.Dropout(dropout)(ff_x)
        x = ff_x + x

    # print(x.shape) # (2,65,1024)
    return x

class Pos_embedding(tf.keras.Model):
    def __init__(self, img_size, patch_size, emb_dim):
        super().__init__(img_size, patch_size, emb_dim)
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.num_patch = (img_size[0]//patch_size[0]) * (img_size[1]//patch_size[1]) # image height -> x.shape[1], image width -> x.shape[2]
        self.pos_emb = tf.Variable(tf.random.normal((1, self.num_patch+1, self.emb_dim), name='pos_emb'))

    def call(self, x):
        b, n, _ = x.shape
        pos_embeddings = repeat(self.pos_emb, '1 c d -> b c d', b=b) # (2,65,1024)
        return pos_embeddings

class Cls_token(tf.keras.Model):
    def __init__(self, img_size, patch_size, emb_dim):
        super().__init__(img_size, patch_size, emb_dim)
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.num_patch = (img_size[0]//patch_size[0]) * (img_size[1]//patch_size[1]) # image height -> x.shape[1], image width -> x.shape[2]
        self.cls_token = tf.Variable(tf.random.normal((1, 1, self.emb_dim)), name='cls_token')

    def call(self, x):
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 c d -> b c d', b=b) # (2,65,1024)
        return cls_tokens

class encoder(tf.keras.Model):
    def __init__(self, img_size, patch_size, emb_dim, depth, heads, mlp_dim, dim_head=64):
        super().__init__(img_size, patch_size, emb_dim, depth, heads, mlp_dim, dim_head)
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.dim_head = dim_head
        # self.num_patch = (img_size[0]//patch_size[0]) * (img_size[1]//patch_size[1]) # image height -> x.shape[1], image width -> x.shape[2]
        # self.pos_embedding = tf.Variable(initial_value=tf.random.normal([1, self.num_patch + 1, emb_dim]))
        # self.cls_token = tf.Variable(initial_value=tf.random.normal([1, 1, emb_dim]))

    def __call__(self, x):
        x = Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=self.patch_size[0], p2=self.patch_size[1])(x) #(2,64,12288)
        x = k.layers.Dense(self.emb_dim/16)(x)
        x = k.layers.Dense(self.emb_dim)(x) # (2,64,1024)
        b, n, _ = x.shape

        # cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=b)
        # pos_embeddings = repeat(self.pos_embedding, '1 n d -> b n d', b=b)
        pos_embeddings = Pos_embedding(img_size=(512,512), patch_size=(64,64), emb_dim=1024)(x) # (2,65,1024)
        cls_tokens = Cls_token(img_size=(512,512), patch_size=(64,64), emb_dim=1024)(x) # (2,1,1024)
        x_vit = k.layers.concatenate([cls_tokens, x], axis=1) # (2,65,1024)
        x_vit = tf.keras.layers.Add()([x_vit, pos_embeddings]) # (2,65,1024)

        vit_layerInfo = []
        for i in range(4):
            x_vit = Transformer(x_vit, cls_tokens, x, b, n, self.emb_dim, self.depth, self.heads, self.dim_head, self.mlp_dim)
            x_vit = k.layers.LayerNormalization()(x_vit)
            x_vit = k.layers.Dense(self.emb_dim)(x_vit)
            vit_layerInfo.append(Rearrange('b c (h w1) -> b h w1 c', w1=int(self.emb_dim**0.5))(x_vit)) # (2,32,32,65)

        return vit_layerInfo
    
def DBUNet(INPUT_SHAPE, OUTPUT_CHANNELS):
    filters = [64, 128, 256, 512]

    input_layer = k.layers.Input(shape=INPUT_SHAPE, batch_size=2, name="input_layer")  # (2,512,512,3)
    vit_layerInfo = encoder(img_size=(512,512), patch_size=(64,64), emb_dim=1024, depth=1, heads=16, mlp_dim=64)(input_layer)
    vit_layerInfo = vit_layerInfo[::-1]
    # (2,32,32,65)

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
    x = Bottleneck(e4, filters[3]) # (2,64,64,512)
    x = k.layers.Dropout(0.1)(x)

    """ Decoder """
    # decoder block 3
    v4 = vit_layerInfo[0]
    v4 = k.layers.Conv2DTranspose(filters=65, kernel_size=2, strides=2, padding='same',
                                  kernel_initializer='he_normal')(v4) # (2,64,64,65)
    d3 = k.layers.concatenate([x, e4, v4], axis=-1) # (2,64,64,833)
    d3 = convT_block(d3, filters[3]) # (2,64,64,256)

    # decoder block 2
    v3 = vit_layerInfo[1]
    v3 = k.layers.Conv2DTranspose(filters=65, kernel_size=2, strides=2, padding='same',
                                  kernel_initializer='he_normal')(v3)
    v3 = k.layers.Conv2DTranspose(filters=65, kernel_size=2, strides=2, padding='same',
                                  kernel_initializer='he_normal')(v3) # (2,128,128,65)
    d2 = k.layers.concatenate([d3, e3, v3], axis=-1) # (2,64,64,577)
    d2 = convT_block(d2, filters[2]) # (2,128,128,128)

    # decoder block 1
    v2 = vit_layerInfo[2]
    v2 = k.layers.Conv2DTranspose(filters=65, kernel_size=2, strides=2, padding='same',
                                  kernel_initializer='he_normal')(v2)
    v2 = k.layers.Conv2DTranspose(filters=65, kernel_size=2, strides=2, padding='same',
                                  kernel_initializer='he_normal')(v2)
    v2 = k.layers.Conv2DTranspose(filters=65, kernel_size=2, strides=2, padding='same',
                                  kernel_initializer='he_normal')(v2) # (2,256,256,65)
    d1 = k.layers.concatenate([d2, e2, v2], axis=-1) # (2,128,128,321)
    d1 = convT_block(d1, filters[1]) # (2,256,256,128)

    # last layer does not have batchnorm and relu
    v1 = vit_layerInfo[3]
    v1 = k.layers.Conv2DTranspose(filters=65, kernel_size=2, strides=2, padding='same',
                                  kernel_initializer='he_normal')(v1)
    v1 = k.layers.Conv2DTranspose(filters=65, kernel_size=2, strides=2, padding='same',
                                  kernel_initializer='he_normal')(v1)
    v1 = k.layers.Conv2DTranspose(filters=65, kernel_size=2, strides=2, padding='same',
                                  kernel_initializer='he_normal')(v1)
    v1 = k.layers.Conv2DTranspose(filters=65, kernel_size=2, strides=2, padding='same',
                                  kernel_initializer='he_normal')(v1) # (2,512,512,65)
    d = k.layers.concatenate([d1, e1, v1], axis=-1) # (2,256,256,193)
    d = conv_block(d, filters[0], n=1) # (2,512,512,64)
    d = conv_block(d, OUTPUT_CHANNELS, n=1, is_bn=False, is_relu=False) # (2,512,512,1)

    if OUTPUT_CHANNELS == 1:
        output = k.activations.sigmoid(d)
    else:
        output = k.activations.softmax(d)

    return tf.keras.Model(inputs=input_layer, outputs=output, name='DBUNet')

# if __name__ == "__main__":
#     input_shape = (512, 512, 3)
#     OUTPUT_CHANNELS = 1
#     model = DBUNet(input_shape, OUTPUT_CHANNELS)
#     metrics = ['accuracy',recall_m,jacard_coef_loss]
#     model.compile(k.optimizers.Adam(learning_rate=0.0001), loss=dice_loss, metrics=metrics)

#     model.summary()