import h5py, pdb, math
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint

import numpy as np
import resnet
import pdb


##############此处改成valid set 的目录
fid = h5py.File('d:/comp_data/training.h5', 'r')
# Loading sentinel-1 data patches
global train_len, valid_len
train_len= fid['sen1'].shape[0]
fid.close()
fid = h5py.File('data/validation.h5', 'r')
valid_len = fid['sen1'].shape[0]
fid.close()

'''
s1 = np.array(fid['sen1'])[:100]
# Loading sentinel-2 data patches
#s2 = np.array(fid['sen2'])[:10000]
# Loading labels
labels = np.array(fid['label'])[:100]
fid.close()
'''

'''
valid_x = np.array(
            np.concatenate(
                (
                    s1,
                    s2
                ),
                axis=3)
        )
'''

#valid_y = labels

print('already load the valid data')

data_idx = 0
batch_size = 512
nb_classes = 17
nb_epoch = 200


def data_generate(data_path, batch_size, schulffe=False):
    fid = h5py.File(data_path, 'r')
    data_len = fid['sen1'].shape[0]
    batch_sum = math.ceil( data_len / batch_size)
    # ceil
    global data_idx
    c = [i for i in range(int(data_len / batch_size))]
    if schulffe:
        np.random.shuffle(c)
    while 1:
        i = data_idx % batch_sum
        y_b = np.array((fid['label'][i * batch_size:(i + 1) * batch_size]))
        x_b = np.array(
            np.concatenate(
                (
                    fid['sen1'][i * batch_size:(i + 1) * batch_size],
                    fid['sen2'][i * batch_size:(i + 1) * batch_size]
                ),
                axis=3)
        )
        #x_b = np.array(fid['sen1'][i * batch_size:(i + 1) * batch_size])
        data_idx += 1
        yield x_b, y_b


lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=10)
csv_logger = CSVLogger('resnet18_competition.csv')
checkpoint  = ModelCheckpoint('./check_file.h5', monitor='val_loss', verbose=0, save_best_only=False,
                                save_weights_only=False, mode='auto', period=1)


#model = resnet.ResnetBuilder.build_resnet_18((18, 32, 32), nb_classes)
model = resnet.ResnetBuilder.build_resnet_18((18, 32, 32), nb_classes)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


##############此处改成train set 的目录
model.fit_generator(data_generate('d:/comp_data/training.h5', batch_size=batch_size),
                        steps_per_epoch = train_len // batch_size,
                        validation_data=data_generate('data/validation.h5', batch_size=batch_size),
                        validation_steps = valid_len // batch_size,
                        epochs=nb_epoch, verbose=1, max_q_size=100,
                        callbacks=[checkpoint,lr_reducer, early_stopper, csv_logger])

model.save('final.h5')