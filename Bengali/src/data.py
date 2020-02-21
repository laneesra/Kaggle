import pyarrow.parquet as pq
import cv2 as cv
import numpy as np
import scipy.misc
#137x236


def read_data():
    test_0 = pq.read_table('../data/train_image_data_0.parquet').to_pandas()
    print('columns', test_0.columns)
    id = test_0.loc[0, 'image_id']
    print('id', id)
    img = np.array(test_0.loc[3, '0':'32331'].values.reshape((137, 236)), np.uint8)
    cv.imshow('image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def plot_distribution(feature, );


if __name__ == '__main__':
    read_data()