from keras.optimizers import Adam

OPTIMIZER = Adam(lr=0.0045)

PARS = {
    'outputs': 1,
    'activation': 'relu',
    'pooling_block': {'trainable': False},
    'information_block': {'convolution': {'simple': 'normalized'}},
    'connection_block': 'not_residual'
}

