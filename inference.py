import model.build_model as modelprovider
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils
import dataset.provider as dataset
import numpy as np


def main():
    # get the data
    data = dataset.read('/home/elias/Nextcloud/1.Masterarbeit/Daten/2020_MA_Elias/ecmwf_*_240.csv')

    # get the shape
    shape_vec, shape_mat, shape_out = dataset.shape(data[1])

    # building the model
    pred2 = np.zeros((11,19))
    pred1 = np.zeros(9)
    #model = modelprovider.build_multi_input_model(pred1.shape, pred2.shape, pred1.shape)
    model = modelprovider.build_multi_input_model(shape_vec, shape_mat, shape_out)

    utils.plot_model(model, "my_model.png", show_shapes=True)

    opt = Adam(lr=1e-3, decay=1e-3 / 200)
    
    # compiling the model
    model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

    # pred = [data[1][0].transpose(), data[1][1]]
    # pred = [data[1][0], data[1][1]]
    # run the model
    prediction = model.predict([pred1[np.newaxis, :],pred2[np.newaxis, :]])
    print(prediction)

if __name__ == "__main__":
    main()