from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as kb


def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    Implement the Jaccard distance metric as a Keras loss function.

    Jaccard = |X & Y| / |X | Y|
            = |X & Y| / (|X| + |Y| - |X & Y|)
            ~ sum(|A*B|) / (sum(|A|) + sum(|B|) - sum(|A*B|))

    The jaccard distance loss is useful for unbalanced data sets. It is based on
    the Dice coefficient but altered to obey the triangle inequality. It has
    also been shifted so it converges on 0 and is smoothed to avoid exploding or
    disappearing gradient.

    Ref: https://en.wikipedia.org/wiki/Jaccard_index
    """
    intersection = kb.sum(kb.abs(y_true * y_pred), axis=-1)
    union = kb.sum(kb.abs(y_true) + kb.abs(y_pred), axis=-1) - intersection
    jac = (intersection + smooth) / (union + smooth)
    return (1 - jac) * smooth


def contracting_block(inputs, out_channels, pool=True, dropout=False):
    """
    Two convolution layers followed by optional dropout/pooling layers.

    The layer immediately preceding the pooling stage is returned as a second
    element of a tuple so that it can be used as a merge layer in the
    expansion blocks.
    """
    block = layers.Conv2D(out_channels, 3,
                           activation='relu',
                           padding='same')(inputs)
    block = layers.Conv2D(out_channels, 3,
                        activation='relu',
                        padding='same')(block)
    if dropout:
        block = layers.Dropout(0.5)(block)
    merge = block
    if pool:
        block = layers.MaxPooling2D(pool_size=(2, 2))(block)
    return block, merge


def expanding_block(inputs, out_channels, merge_layer=None):
    """Deconvolution layer followed by merge and two convolution layers."""
    block = layers.UpSampling2D(size = (2, 2))(inputs)
    block = layers.Conv2D(out_channels, 3,
                           activation='relu',
                           padding='same')(block)
    if not merge_layer is None:
        block = layers.concatenate([merge_layer, block], axis=3)
    block = layers.Conv2D(out_channels, 3,
                        activation='relu',
                        padding='same',
                        kernel_initializer='he_normal')(block)
    return block


def unet(input_size=(1024, 1024, 3), weights=None):
    """
    Compile (and  optionally load weights for) a U-Net model with Jaccard loss.

    The U-Net architecture is a symmetric neural network consisting of a
    contracting and expanding path, allowing for both the capture of context
    and precise localisation. The model can either be compiled with randomly
    distributed initial weights or a weights object can be optionally passed
    in. Merge layers are used to allow knowledge to be transfered directly from
    the contracting path to the expanding path.

    Ref: https://arxiv.org/abs/1505.04597
    """
    inputs = layers.Input(input_size)

    # contraction path
    contract_1, merge_1 = contracting_block(inputs, 16)
    contract_2, merge_2 = contracting_block(contract_1, 32)
    contract_3, merge_3 = contracting_block(contract_2, 64)
    contract_4, merge_4 = contracting_block(contract_3, 128, dropout=True)
    contract_5, _ = contracting_block(contract_4, 256, pool=False,
                                      dropout=True)

    # expansion path
    expand_1 = expanding_block(contract_5, 128, merge_4)
    expand_2 = expanding_block(expand_1, 64, merge_3)
    expand_3 = expanding_block(expand_2, 32, merge_2)
    expand_4 = expanding_block(expand_3, 16, merge_1)

    # output
    # out_1 = layers.Conv2D(2, 3,
    #                      activation='relu',
    #                      padding='same')(expand_4)

    out_2 = layers.Conv2D(1, 1, activation='sigmoid')(expand_4)

    # compile model
    model = Model(inputs = inputs, outputs = out_2)
    model.compile(optimizer = Adam(lr=1e-3),
                  loss='binary_crossentropy', #jaccard_distance_loss
                  metrics=['accuracy'])

    # load weights if given
    if weights:
        model.load_weights(weights)

    return model
