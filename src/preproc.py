# preproc.py
import os
import config

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import sklearn.model_selection as model_selection

def make_padding(image, output_size = 36):
    input_size, _ , channel = image.shape
    miss = int((output_size - input_size)/2)
    b1 = np.ones((miss ,input_size, channel))
    b2 = np.ones((output_size, miss, channel))
    rect = np.concatenate(
        (np.concatenate((b1,image),axis = 0),b1),
        axis = 0
        )
    padd_image = np.concatenate(
        (np.concatenate((b2,rect),axis = 1),b2),
        axis = 1
        )
    return padd_image

def image_to_vector(image: np.ndarray) -> np.ndarray:
    length, height, depth = image.shape
    return image.reshape((length * height * depth, 1))


def build_dataset():
    im_files_circles = [
        config.CIRCLES  +  im_circ for im_circ in os.listdir(config.CIRCLES)
        ]
    im_files_squares = [
        config.SQUARES  +  im_squer for im_squer in os.listdir(config.SQUARES)
        ]
    im_files_triangles = [
        config.TRIANGLES  +  im_tria for im_tria in os.listdir(config.TRIANGLES)
        ]

    images_plt_circles = [
        plt.imread(f) for f in im_files_circles
        ]
    images_plt_squares = [
        plt.imread(f) for f in im_files_squares
        ]
    images_plt_triangles = [
        plt.imread(f) for f in im_files_triangles
        ]

    images_circles = np.array(images_plt_circles)
    images_squares = np.array(images_plt_squares)
    images_triangles = np.array(images_plt_triangles)

    circles = [np.array(circ) for circ in images_circles]
    squares = [np.array(sque) for sque in images_squares]
    triangles = [np.array(tri) for tri in images_triangles]

    circles_1D = np.array([image_to_vector(c) for c in circles])
    triangles_1D = np.array([image_to_vector(t) for t in triangles])
    squares_1D = np.array([image_to_vector(s) for s in squares])
    
    
    df_circles = pd.DataFrame(
        data=circles_1D.reshape(circles_1D.shape[0], circles_1D.shape[1])
        )
    df_triangles = pd.DataFrame(
        data=triangles_1D.reshape(triangles_1D.shape[0], triangles_1D.shape[1])
        )
    df_squares = pd.DataFrame(
        data=squares_1D.reshape(squares_1D.shape[0], squares_1D.shape[1])
        )
    
    df = pd.concat(
        [df_circles, df_triangles, df_squares],
        axis=0
        )
    
    targets = pd.Series(
        data=['circle' for c in range(100)] \
            + ['triangle' for t in range(100)] \
                + ['squares' for s in range(100)],
        dtype='category'
        )
    
    df['target'] = targets.values
    df.target = df.target.cat.codes.values

    df_train, df_valid = model_selection.train_test_split(
        df, test_size=0.2, random_state=config.SEED
        )


    df_train.to_csv(config.INPUT_FOLD + "train_dataset.csv")
    df_valid.to_csv(config.INPUT_FOLD + "valid_dataset.csv")

    with open("config.py", "r") as f:
        data = f.readlines()
        with open("config.py","w") as file:
            for line in data:
                file.write(line)
            file.write("""TRAIN_DATA = "../input/train_dataset.csv" """+"\n")
            file.write("""VALID_DATA = "../input/valid_dataset.csv" """)
            file.close()
        f.close()

    
if __name__ == "__main__":
    build_dataset()