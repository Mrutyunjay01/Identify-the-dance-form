import os
import cv2
import argparse

'''
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--directory', required=True,
                help='Path to image directory')
args = vars(ap.parse_args())

traindf = pd.read_csv('../dataset/train.csv')
testdf = pd.read_csv('../dataset/test.csv')
'''


def sort_filanames(dataset):
    """

    :param dataset: dataframe containing filenames and target values
    :return:
    """

    # concatenate the '.jpg' part of the filenames to sort it
    dataset['Image'] = dataset['Image'].apply(lambda x: x[: x.find('.')])

    # print(dataset.head())
    dataset = dataset.sort_values('Image')

    dataset = dataset.reset_index(drop=True)

    return dataset
    pass


def image_loder(folderPath, dataset):
    images = []

    for i, filename in enumerate(os.listdir(folderPath)):

        assert filename == dataset.Image[i] + '.jpg'
        img = cv2.imread(os.path.join(folderPath, filename))

        if img is not None:
            images.append(img)
    return images
