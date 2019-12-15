import numpy as np
import os
from skimage.transform import resize
from skimage.io import imsave, imread
import warnings

from u_model import get_unet, IMG_COLS as img_cols, IMG_ROWS as img_rows
from data import load_train_data, load_test_data, load_nerve_presence, load_test_ids
from configuration import PARS, OPTIMIZER
from keras.callbacks import ModelCheckpoint, EarlyStopping


def preprocess(imgs, to_rows=None, to_cols=None):
    if to_rows is None or to_cols is None:
        to_rows = img_rows
        to_cols = img_cols

    print(imgs.shape)
    imgs_p = np.ndarray((imgs.shape[0], to_rows, to_cols, imgs.shape[3]), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i, :, :, 0] = resize(imgs[i, :, :, 0], (to_rows, to_cols), preserve_range=True)
    return imgs_p


# 학습 및 예측 함수
def train_and_predict():
    print('-' * 40)
    print('Loading and preprocessing train data...')
    print('-' * 40)
    imgs_train, imgs_mask_train = load_train_data()	# 기존 train 데이터셋 불러오기
    imgs_present = load_nerve_presence()

    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    # 이미지 센터링 및 표준화 작업
    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)
    std = np.std(imgs_train)
    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to be in {0, 1} instead of {0, 255}

    print('-' * 40)
    print('Creating and compiling model...')
    print('-' * 40)

    # load model - the Learning rate scheduler choice is most important here
    model = get_unet(optimizer=OPTIMIZER, pars=PARS)

    # model checkpoint : validation error를 모니터링하면서, 이전 epoch에 비해 validation performance가 높았던 모델을 반환하는 변수.
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)
    # early stopping : 조기 종료 기법. 이전 patience만큼 epoch 때와 비교해서 오차가 증가했다면 학습을 중단하는 변수.
    early_stopping = EarlyStopping(patience=5, verbose=1) # patience : 오차를 보기 위해 과거 몇 epoch까지 거슬러 올라갈 것인지 정하는 인자.

    print('-' * 40)
    print('Fitting model...')
    print('-' * 40)

    if PARS['outputs'] == 1:
        imgs_labels = imgs_mask_train
    else:
        imgs_labels = [imgs_mask_train, imgs_present]

    # 학습 모델 환경 값 설정.
    model.fit(imgs_train, imgs_labels,
              batch_size=128, epochs=50,
              verbose=1, shuffle=True,
              validation_split=0.2,
              callbacks=[model_checkpoint, early_stopping]) # 위에서 설정한 model checkpoint와 early stopping 변수를 모델에서 콜백으로 사용하기 위함.

    print('-' * 40)
    print('Loading and preprocessing test data...')
    print('-' * 40)
    imgs_test = load_test_data()
    imgs_id_test = load_test_ids()
    imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    print('-' * 40)
    print('Loading saved weights...')
    print('-' * 40)
    model.load_weights('weights.h5')

    print('-' * 40)
    print('Predicting masks on test data...')
    print('-' * 40)

    imgs_mask_test = model.predict(imgs_test, verbose=1)

    if PARS['outputs'] == 1:
        np.save('imgs_mask_test.npy', imgs_mask_test)
    else:
        np.save('imgs_mask_test.npy', imgs_mask_test[0])
        np.save('imgs_mask_test_present.npy', imgs_mask_test[1])

    pred_dir = 'preds'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for image, image_id in zip(imgs_mask_test, imgs_id_test):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            imsave(os.path.join(pred_dir, str(image_id) + '_pred.png'), image)
    
    print('----------------------------------------')
    print('Saving Predicted Finished.')
# --------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    train_and_predict()
