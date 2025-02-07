import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import ResNet50
from myconfig import MyConfig


def conv_block(inputs, filters, kernel_size=3, activation='relu', padding='same'):
    x = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer=K.initializers.he_normal, kernel_regularizer=K.regularizers.l2(5e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    return x


def residual_block(inputs, filters, kernel_size=3, activation='relu', padding='same'):
    x = conv_block(inputs, filters, kernel_size, activation, padding)
    x = conv_block(x, filters, kernel_size, activation, padding)
    shortcut = layers.Conv2D(filters, kernel_size=1, padding=padding, kernel_initializer=K.initializers.he_normal, kernel_regularizer=K.regularizers.l2(5e-4))(inputs)
    shortcut = layers.BatchNormalization()(shortcut)
    x = layers.add([x, shortcut])
    x = layers.Activation(activation)(x)
    return x


def upsample_block(inputs, filters, kernel_size=2, strides=2, padding='same'):
    x = layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding, kernel_initializer=K.initializers.he_normal, kernel_regularizer=K.regularizers.l2(5e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x


def ResUNet(input_shape, num_classes):
    input = Input(input_shape)

    inputs = input
    if input_shape[2] == 1:
        inputs = layers.Concatenate()([input, input, input])

    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)

    e1 = base_model.get_layer("conv1_relu").output
    e2 = base_model.get_layer("conv2_block3_out").output
    e3 = base_model.get_layer("conv3_block4_out").output
    e4 = base_model.get_layer("conv4_block6_out").output
    e5 = base_model.get_layer("conv5_block3_out").output

    d1 = upsample_block(e5, 512)
    d1 = layers.concatenate([d1, e4])
    d1 = residual_block(d1, 512)

    d2 = upsample_block(d1, 256)
    d2 = layers.concatenate([d2, e3])
    d2 = residual_block(d2, 256)

    d3 = upsample_block(d2, 128)
    d3 = layers.concatenate([d3, e2])
    d3 = residual_block(d3, 128)

    d4 = upsample_block(d3, 64)
    d4 = layers.concatenate([d4, e1])
    d4 = residual_block(d4, 64)

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(d4)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(input, outputs)
    return model

if __name__ == '__main__':
    myconfig = MyConfig()
    input_shape = myconfig.input_shape
    num_classes = myconfig.class_num

    model = ResUNet(input_shape, num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()