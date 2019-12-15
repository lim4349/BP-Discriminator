import numpy as np
from keras.layers import Input, concatenate, Conv2D, UpSampling2D, Dense
from keras.layers import Dropout, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K

from metric import dice_coef, dice_coef_loss
from u_model_blocks import pooling_block, connection_block, information_block
from configuration import PARS


IMG_ROWS, IMG_COLS = 80, 112  # 이미지의 ROW 줄과 COLUMN 줄. 
K.set_image_data_format('channels_last')  # (number of images, rows per image, cols per image, channels)


def get_unet_customised(optimizer, pars=PARS):
    # string, activation function
    activation = pars.get('activation')
    #activation = 'relu'
    # input
    inputs = Input((IMG_ROWS, IMG_COLS, 1), name='main_input')
    print('inputs:', inputs._keras_shape)

    #
    # U-net에서 내려가기 시작
    #

    # 1층
    conv1 = information_block(inputs, 32, padding='same', activation=activation, pars=pars)
    print('conv1', conv1._keras_shape)
    pool1 = pooling_block(inputs=conv1, filters=32, activation=activation, pars=pars)
    print('pool1', pool1._keras_shape)
    pool1 = Dropout(0.5)(pool1)
    print('pool1', pool1._keras_shape)

    conv2 = information_block(pool1, 64, padding='same', activation=activation, pars=pars)
    print('conv2', conv2._keras_shape)
    pool2 = pooling_block(inputs=conv2, filters=64, activation=activation, pars=pars)
    print('pool2', pool2._keras_shape)
    pool2 = Dropout(0.5)(pool2)
    print('pool2', pool2._keras_shape)

    conv3 = information_block(pool2, 128, padding='same', activation=activation, pars=pars)
    print('conv3', conv3._keras_shape)
    pool3 = pooling_block(inputs=conv3, filters=128, activation=activation, pars=pars)
    print('pool3', pool3._keras_shape)
    pool3 = Dropout(0.5)(pool3)
    print('pool3', pool3._keras_shape)

    conv4 = information_block(pool3, 256, padding='same', activation=activation, pars=pars)
    print('conv4', conv4._keras_shape)
    pool4 = pooling_block(inputs=conv4, filters=256, activation=activation, pars=pars)
    print('pool4', pool4._keras_shape)
    pool4 = Dropout(0.5)(pool4)
    print('pool4', pool4._keras_shape)

    #
    # bottom level of the U-net
    # 가운데 층. 
    conv5 = information_block(pool4, 512, padding='same', activation=activation, pars=pars)
    print('conv5', conv5._keras_shape)
    conv5 = Dropout(0.5)(conv5)    # 드랍아웃. 오버피팅 문제를 해결하고자, 일반화 성능을 향상시키기 위한 방법.
    print('conv5', conv5._keras_shape)

    #
    # auxiliary output for predicting probability of nerve presence
    #
    if pars['outputs'] == 2:
        pre = Conv2D(1, kernel_size=(1, 1), kernel_initializer='he_normal', activation='sigmoid')(conv5)	# 
        pre = Flatten()(pre)
        aux_out = Dense(1, activation='sigmoid', name='aux_output')(pre)

    #
    # up the U-net
    # U-net 가운데 최하층에서 올라가기 시작
    #

    after_conv4 = connection_block(conv4, 256, padding='same', activation=activation,
                                   pars=pars)
    print('after_conv4', after_conv4._keras_shape)
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), after_conv4], axis=3)
    conv6 = information_block(up6, 256, padding='same', activation=activation, pars=pars)
    print('conv6', conv6._keras_shape)
    conv6 = Dropout(0.5)(conv6)
    print('conv6', conv6._keras_shape)

    after_conv3 = connection_block(conv3, 128, padding='same', activation=activation,
                                   pars=pars)
    print('after_conv3', after_conv3._keras_shape)
    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), after_conv3], axis=3)
    conv7 = information_block(up7, 128, padding='same', activation=activation, pars=pars)
    print('conv7', conv7._keras_shape)
    conv7 = Dropout(0.5)(conv7)
    print('conv7', conv7._keras_shape)

    after_conv2 = connection_block(conv2, 64, padding='same', activation=activation, pars=pars)
    print('after_conv2', after_conv2._keras_shape)
    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), after_conv2], axis=3)
    conv8 = information_block(up8, 64, padding='same', activation=activation, pars=pars)
    print('conv8', conv8._keras_shape)
    conv8 = Dropout(0.5)(conv8)
    print('conv8', conv8._keras_shape)

    after_conv1 = connection_block(conv1, 32, padding='same', activation=activation, pars=pars)
    print('after_conv1', after_conv1._keras_shape)
    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), after_conv1], axis=3)
    conv9 = information_block(up9, 32, padding='same', activation=activation, pars=pars)
    print('conv9', conv9._keras_shape)
    conv9 = Dropout(0.5)(conv9)
    print('conv9', conv9._keras_shape)

    # main output
    conv10 = Conv2D(1, kernel_size=(1, 1), kernel_initializer='he_normal', activation='sigmoid', name='main_output')(
        conv9)
    print('conv10', conv10._keras_shape)

    # creating a model
    # compiling the model
    if pars['outputs'] == 1:
        model = Model(inputs=inputs, outputs=conv10)
        model.compile(optimizer=optimizer,
                      loss={'main_output': dice_coef_loss},
                      metrics={'main_output': dice_coef})
    else:
        model = Model(inputs=inputs, outputs=[conv10, aux_out])
        model.compile(optimizer=optimizer,
                      loss={'main_output': dice_coef_loss, 'aux_output': 'binary_crossentropy'},
                      metrics={'main_output': dice_coef, 'aux_output': 'acc'},
                      loss_weights={'main_output': 1., 'aux_output': 0.5})

    return model


# ----------------------------------------------------------------------------------------------------------------------

get_unet = get_unet_customised

if __name__ == '__main__':


    img_rows = IMG_ROWS
    img_cols = IMG_COLS
    model = get_unet(Nadam(lr=1e-5), pars=PARS)

    x = np.random.random((1, img_rows, img_cols, 1))
    result = model.predict(x, 1)
    print(result)
    print('params', model.count_params())
    print('layer num', len(model.layers))
