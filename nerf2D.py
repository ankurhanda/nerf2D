from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Dense, BatchNormalization 
from tensorflow.keras import Model

from scipy import signal, special 
from PIL import Image
import cv2 as cv2


class PositionEncoding(object):
    def __init__(self, image_np, basis_function):
        super().__init__()
        
        self.dataset_size = []

        H, W, C = image_np.shape
    
        self.dataset = [np.array([-1, -1, -1, -1, -1])]*W*H

        L = 10

        x_linspace = (np.linspace(0, W-1, W)/W)*2 -1 
        y_linspace = (np.linspace(0, H-1, H)/H)*2 -1

        x_el = []
        y_el = []

        x_el_hf = []
        y_el_hf = []

        self.basis_function = basis_function
        
        # cache the values so you don't have to do function calls at every pixel
        for el in range(0, L):
            val = 2 ** el 
            
            if basis_function == 'rbf':
        
                # Trying Random Fourier Features https://www.cs.cmu.edu/~schneide/DougalRandomFeatures_UAI2015.pdf
                # and https://gist.github.com/vvanirudh/2683295a198a688ef3c49650cada0114

                # Instead of a phase shift of pi/2, we could randomise it [-pi, pi]

                M_1 = np.random.rand(2,2)

                phase_shift = np.random.rand(1) * np.pi

                x_1_y_1 = np.sin(val * np.matmul(M_1, np.vstack((x_linspace, y_linspace))))
                x_el.append(x_1_y_1[0,: ])
                y_el.append(x_1_y_1[1,: ])

                x_1_y_1 = np.sin(val * np.matmul(M_1, np.vstack((x_linspace, y_linspace))) + phase_shift)
                x_el_hf.append(x_1_y_1[0,: ])
                y_el_hf.append(x_1_y_1[1,: ])

            elif basis_function == 'diric':

                x = special.diric(np.pi * x_linspace, val)
                x_el.append(x)

                x = special.diric(np.pi * x_linspace + np.pi/2.0, val)
                x_el_hf.append(x)

                y = special.diric(np.pi * y_linspace, val)
                y_el.append(y)

                y = special.diric(np.pi * y_linspace + np.pi/2.0, val)
                y_el_hf.append(y)
            
            elif basis_function == 'sawtooth':
                x = signal.sawtooth(val * np.pi * x_linspace)
                x_el.append(x)

                x = signal.sawtooth(val * np.pi * x_linspace + np.pi/2.0)
                x_el_hf.append(x)

                y = signal.sawtooth(val * np.pi * y_linspace)
                y_el.append(y)

                y = signal.sawtooth(val * np.pi * y_linspace + np.pi/2.0)
                y_el_hf.append(y)

            elif basis_function == 'sin_cos':
                
                x = np.sin(val * np.pi * x_linspace)
                x_el.append(x)

                x = np.cos(val * np.pi * x_linspace)
                x_el_hf.append(x)

                y = np.sin(val * np.pi * y_linspace)
                y_el.append(y)

                y = np.cos(val * np.pi * y_linspace)
                y_el_hf.append(y)

        # TODO: vectorise this code!
        for y_i in range(0, H):
            for x_i in range(0, W):

                r, g, b = image_np[y_i, x_i]

                p_enc = []

                # i.e. passing raw coordinates instead of positional encoding 
                if basis_function == 'raw_xy':

                    xdash = (x_i/W)*2 -1
                    ydash = (y_i/H)*2 -1
                    p_enc = [xdash, ydash]

                else:

                    for li in range(0, L):

                        p_enc.append(x_el[li][x_i])
                        p_enc.append(x_el_hf[li][x_i])

                        p_enc.append(y_el[li][y_i])
                        p_enc.append(y_el_hf[li][y_i])

                p_enc = p_enc + [x_i, y_i, r*2 -1, g*2 -1, b*2 -1]

                self.dataset[y_i * W + x_i]  = np.array(p_enc)

        self.dataset_size = len(self.dataset)
        print('size of dataset_size = ', self.dataset_size)

        self.ind = np.arange(np.sum(self.dataset_size))
        np.random.shuffle(self.ind)

        self.batch_count = 0

    def get_batch(self, batch_size=10):

        input_vals = []
        output_vals = []
        indices_vals = []
        
        for i in range(batch_size):
        
            if self.batch_count * batch_size + i >= self.dataset_size:
                self.batch_count = 0
                np.random.shuffle(self.ind)
                print('************************************************* new shuffle *****************************************')
                # break
        
            p_enc = self.dataset[self.ind[self.batch_count * batch_size + i]]

            input_vals.append(p_enc[0:-5])

            r, g, b = p_enc[-3], p_enc[-2], p_enc[-1]
            x, y = p_enc[-5], p_enc[-4]

            output_vals.append([r, g, b])

            indices_vals.append([x, y])

        self.batch_count += 1 
        return np.array(input_vals), np.array(output_vals), np.array(indices_vals)

im = Image.open('dataset/glasses.jpg')
im2arr = np.array(im) 

testimg = im2arr 
testimg = testimg / 255.0  
H, W, C = testimg.shape

PE = PositionEncoding(testimg, 'sin_cos')
dataset_size = PE.dataset_size

def build_model(output_dims=3):
    model = tf.keras.Sequential([
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(128, activation='relu'),
        
        BatchNormalization(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(128, activation='relu'),

        BatchNormalization(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(128, activation='relu'),

        BatchNormalization(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(128, activation='relu'),

        Dense(output_dims, activation='linear')
    ])
    return model

loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(lr=1e-2)
EPOCHS = 5

model = build_model(output_dims=3)

batch_size = 1024
decay = 0.999

count = 0 
epoch_no = 0 

_read_img = np.zeros((H, W, 3))

save_every = 200

while True:

        inp_batch, inp_target, ind_vals = PE.get_batch(batch_size=batch_size)

        with tf.GradientTape() as tape:

            output = model(inp_batch, training=True)

            loss_map = tf.sqrt(loss_object(output, inp_target))

            if count > 0 and count % save_every == 0:

                inp_batch, inp_target, ind_vals = PE.get_batch(batch_size=dataset_size)
                output = model(inp_batch, training=False)

                ind_vals_int = ind_vals.astype('int')
                ind_vals_int = ind_vals_int[:, 1] * W + ind_vals_int[:, 0]

                np.put(_read_img[:, :, 0], ind_vals_int, np.clip((output[:, 0]+1)/2.0, 0, 1))
                np.put(_read_img[:, :, 1], ind_vals_int, np.clip((output[:, 1]+1)/2.0, 0, 1))
                np.put(_read_img[:, :, 2], ind_vals_int, np.clip((output[:, 2]+1)/2.0, 0, 1))

                fileName = 'training_evolution_' + PE.basis_function + '_{:04d}.jpg'.format(int(epoch_no))
                save_img = np.copy(_read_img[...,::-1]*255.0)
                cv2.imwrite(fileName, save_img.astype('uint8'))

                # run only or 1000 epochs
                if epoch_no > 1000:
                    break


            cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Align Example', _read_img[...,::-1])
            key = cv2.waitKey(1)

            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
   
            print('loss = {}, learning_rate= {}, batch_no = {}, epoch = {}, batches_per_epoch = {}'.format(loss_map.numpy(), 
                                                                                            optimizer.learning_rate.numpy(),
                                                                                            count,
                                                                                            epoch_no,
                                                                                            batch_size))
        gradients = tape.gradient(loss_map, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
        count += 1 

        if PE.batch_count == 1 and count > 1:
            # lr = float(tf.keras.backend.get_value(optimizer.lr))
            # tf.keras.backend.set_value(optimizer.lr, lr * 0.99)
            epoch_no += 1 