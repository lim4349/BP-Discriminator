import os
import numpy as np
from skimage.io import imread

_dir = os.path.abspath('')
os.chdir(_dir)

data_path = os.path.join('input/ultrasound-nerve-segmentation', '')
preprocess_path = os.path.join(_dir, 'np_data')

if not os.path.exists(preprocess_path):
    os.mkdir(preprocess_path)
print(os.listdir(_dir))

# train data
img_train_path = os.path.join(preprocess_path, 'imgs_train.npy')
img_train_mask_path = os.path.join(preprocess_path, 'imgs_mask_train.npy')
img_train_patients = os.path.join(preprocess_path, 'imgs_patient.npy')
img_nerve_presence = os.path.join(preprocess_path, 'nerve_presence.npy')

# test data
img_test_path = os.path.join(preprocess_path, 'imgs_test.npy')
img_test_id_path = os.path.join(preprocess_path, 'imgs_id_test.npy')

# image dimensions
image_rows = 420
image_cols = 580

def load_test_data():
    print('Loading test data from %s' % img_test_path)
    imgs_test = np.load(img_test_path)
    return imgs_test


def load_test_ids():
    print('Loading test ids from %s' % img_test_id_path)
    imgs_id = np.load(img_test_id_path)
    return imgs_id


def load_train_data():
    print('Loading train data from %s and %s' % (img_train_path, img_train_mask_path))
    imgs_train = np.load(img_train_path)
    imgs_mask_train = np.load(img_train_mask_path)
    return imgs_train, imgs_mask_train


def load_patient_num():
    print('Loading patient numbers from %s' % img_train_patients)
    return np.load(img_train_patients)


def load_nerve_presence():
    print('Loading nerve presence array from %s' % img_nerve_presence)
    return np.load(img_nerve_presence)


def get_patient_nums(string):
    patient, photo = string.split('_')
    photo = photo.split('.')[0]
    return int(patient), int(photo)


def get_nerve_presence(mask_array):
    print("type(mask_array):", type(mask_array))
    print("mask_array.shape:", mask_array.shape)
    return np.array([int(np.sum(mask_array[i, :, :, 0]) > 0) for i in range(mask_array.shape[0])])


def create_train_data():
    train_data_path = os.path.join(data_path, 'train')
    images = os.listdir(train_data_path)
    total = len(images) // 2

    imgs = np.ndarray((total, image_rows, image_cols, 1), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_cols, 1), dtype=np.uint8)
    i = 0
    print('Creating training images...')
    img_patients = np.ndarray((total,), dtype=np.uint8)
    for image_name in images:
        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.tif'
        patient_num = image_name.split('_')[0]
        img = imread(os.path.join(train_data_path, image_name), as_gray=True)
        img_mask = imread(os.path.join(train_data_path, image_mask_name), as_gray=True)

        imgs[i, :, :, 0] = img
        imgs_mask[i, :, :, 0] = img_mask
        img_patients[i] = patient_num
        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    # saving patient numbers, train images, train masks, nerve presence
    np.save(img_train_patients, img_patients)
    np.save(img_train_path, imgs)
    np.save(img_train_mask_path, imgs_mask)
    np.save(img_nerve_presence, get_nerve_presence(imgs_mask))

    print('Saving to .npy files done.')


def create_test_data():
    test_data_path = os.path.join(data_path, 'test')
    images = os.listdir(test_data_path)
    total = len(images)

    imgs = np.ndarray((total, image_rows, image_cols, 1), dtype=np.uint8)
    imgs_id = np.ndarray((total,), dtype=np.int32)

    i = 0
    print('Creating test images...')
    for image_name in images:
        img_id = int(image_name.split('.')[0])
        img = imread(os.path.join(test_data_path, image_name), as_gray=True)

        imgs[i, :, :, 0] = img
        imgs_id[i] = img_id

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save(img_test_path, imgs)
    np.save(img_test_id_path, imgs_id)
    print('Saving to .npy files done.')


# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    create_train_data()
    create_test_data()


print(os.listdir(preprocess_path))
