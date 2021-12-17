import numpy as np

from src.common.__commonFuns__ import *
# import keras
from keras import callbacks
from keras import losses
from keras import Input, Model
from keras import backend as K
from tqdm.keras import TqdmCallback
from sklearn.model_selection import train_test_split
from plot_keras_history import plot_history

# tensorflow
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.keras.utils import plot_model
from tensorflow import keras
from tensorflow.keras import layers

# (device_lib.list_local_devices())


# ------------------- AUTOENCODERS

'''
Variational autoencoder class that takes from Keras layers and allows to 
give two layer input at a time, calculating z_mean and z_log_var
'''


class Sampling(layers.Layer):
    def call(self, inputs, **kwargs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

"""
variational autoencoder layers creator 
- parameter alfa allows to specify the leakage of leaky ReLu functions
- nlayers specifies the depth of the autoencoder
- latent_dim is the width of the most inner layer
- shape is the shape of the encoder layer input and the output of the decoder
"""
def autoEnc(latent_dim, shape, nlayers=1, alfa=0.1, batch = 30):
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(batch, latent_dim),
                                  mean=0., stddev=1.0)
        return z_mean + K.exp(z_log_var) * epsilon

    print(f"creating {nlayers} with input shape {shape}")
    input_size = shape[1]
    compression = int(input_size * latent_dim / input_size)
    reduction = compression // nlayers
    if input_size - (nlayers - 1) * reduction <= 0:
        print(f"to many layers {nlayers} for compression {compression}")

    # ------ encoder
    inputer = Input(shape)
    encoder = None
    if nlayers == 1:
        # shap = (None
        encoder = layers.Dense(input_size - (nlayers - 1) * reduction,
                               activation=keras.activations.sigmoid)(inputer)
        # we must have size bigger than 0 so check maximum for those number of layers
    else:
        # add linear layer
        encoder = layers.Dense(input_size - reduction,
                               activation=layers.LeakyReLU(alpha=alfa))(inputer)
        # add leaky relus
        counter = 2
        for i in range(nlayers - 3):
            encoder = layers.Dense(input_size - counter * reduction,
                                   activation=layers.LeakyReLU(alpha=alfa))(encoder)
            counter += 1

        # add last dense layer
        encoder = layers.Dense(input_size - (nlayers - 1) * reduction,
                               activation=keras.activations.sigmoid)(encoder)

    # --------- Latent log variance and mu layers
    fc_logvar = layers.Dense(latent_dim, name="log_var")(encoder)
    fc_mu = layers.Dense(latent_dim, name="mean")(encoder)
    #z = Sampling()([fc_mu, fc_logvar])
    z = layers.Lambda(sampling)([fc_mu, fc_logvar])

    encod = keras.Model(inputer, [fc_mu, fc_logvar, z], name="encoder")
    #encod.summary()

    # ------ decoder
    inputer2 = keras.Input(latent_dim)
    decoder = None
    #if nlayers == 1:
    #    decoder = layers.Dense(input_size, activation=keras.activations.sigmoid)(inputer2)
    #    # we must have size bigger than 0 so check maximum for those number of layers
    #else:
    #    counter = nlayers - 1
    #    decoder = layers.Dense(input_size - counter * reduction, activation=layers.LeakyReLU(alpha=alfa))(inputer2)

    #    counter -= 1
    #    for i in range(nlayers - 2):
    #        decoder = layers.Dense(input_size - counter * reduction, activation=layers.LeakyReLU(alpha=alfa))(decoder)
    #        counter -= 1

        # add last dense layer
    #    decoder = layers.Dense(input_size, activation=keras.activations.sigmoid)(decoder)
    decoder_h = layers.Dense(latent_dim, activation='relu')
    decoder_mean = layers.Dense(input_size, activation='sigmoid')
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)


    #decoder_h = layers.Dense(latent_dim, activation='relu')
    #decod = decoder_h(z)
    #decod = decoder(decod)
    # return model
    #decod = keras.Model(inputer, decod)
    decod_M = keras.Model(inputer, x_decoded_mean, name="decoder")
    #decod.summary()
    #return decod
    return encod, decod_M


'''
Variational autoencoder class
'''


class VAE(keras.Model):
    def __init__(self, encoder, decoder, epochs, **kwargs):
        super(VAE, self).__init__(**kwargs)
        # LOG AND MEAN ARE IN THE ENCODER ALREADY
        self.epochs = epochs
        self.epoch = 0
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, mydata):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, _ = self.encoder(mydata)
            # print("z : =", z)
            reconstruction = self.decoder(mydata)
            # print("data : =", mydata)
            # print("reconstruction : =", reconstruction)
            reconstruction_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False,
                                                                     reduction=tf.keras.losses.Reduction.SUM)
            reconstruction_loss = tf.reduce_mean(reconstruction_loss(y_true=mydata, y_pred=reconstruction))
            kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)


            weight = 0.85 * (self.epoch / self.epochs)
            #kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            #kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss * weight  # * self.weight

            # print(f"with weight = {self.weight} and {total_loss}")
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.total_loss_tracker.update_state(total_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def scheduler(self, epoch, lr):
        self.epoch = epoch
        # print(self.weight)
        return lr  # self.weight

    def call(self, x, **kwargs):
        #x = self.encoder(x)
        return self.decoder(x)


'''
Function to create the learning of the wavevectors distribution
'''


def fileAutoencode(directory, model, latent_dim, epo,
                   layer_num, trainsize=0.8,
                   filenum=None, batch=10, verbose=2,
                   activation='relu', trainAll = False,
                   save=True, savefiles=False):
    folderLog = directory + "_" + model.getInfo() + kPSep
    folder = folderLog + "wavefunctions" + kPSep
    folderSaveNew = folderLog + "wavefunctions_encoder" + kPSep
    createFolder([folderSaveNew])
    savename = folderLog + f'myModel_latent={latent_dim / model.N:.3f}'
    # read files
    wavefuns = []

    files = list(filter(lambda x: x.endswith('.dat'), os.listdir(folder)))
    maximum = len(files)
    if filenum is not None:
        maximum = filenum

    counter = 0
    # create squares of the wavefunctions
    for filename in files:
        ''''''
        tmp = np.square(np.genfromtxt(folder + filename))
        wavefuns.append(tmp)
        if counter == maximum:
            break
        counter += 1

    # separate training etc
    data_train = np.array(wavefuns[0:int(trainsize*maximum)])
    data_test = np.array(wavefuns[int(trainsize*maximum):])
    tmp = np.concatenate([data_train,data_test])
    # print(data_test)

    print(f'\n\n\t\tMaking autoencoder with latent={latent_dim / model.N}\t\t\n\n')
    encoder, decoder = autoEnc(latent_dim, (None, int(model.N)), layer_num, alfa=0.15)
    vae = VAE(encoder, decoder, epo)
    # first model compile
    vae.compile(optimizer=keras.optimizers.Adam(lr=1e-3))

    # making callbacks
    early_stopping_cb = callbacks.EarlyStopping(monitor="loss", patience=3)
    callback = [early_stopping_cb]
    if save:
        callback.append(keras.callbacks.ModelCheckpoint(savename + ".h5", save_best_only=True))
    if verbose == 2:
        callback.append(TqdmCallback(verbose=verbose))

    if trainAll:
        history = vae.fit(tmp,
                        epochs=epo,
                        shuffle=True,
                        batch_size=batch,
                        verbose=0,
                        callbacks=callback)
        plot_history(history, path=savename + ',training.png')
    else:
        # make artificial history
        history = {}
        history['history'] = {}
        history['history']['total_loss'] = []
        history['history']['reconstruction_loss'] = []
        history['history']['kl_loss'] = []

        tot_loss, reconstr_loss, kl_loss = 0, 0, 0
        for ep in range(epo):
            vae.epoch = ep
            vae.compile(optimizer=keras.optimizers.Adam(lr=1e-3))
            # train iterations
            story = vae.fit(tmp,
                            #validation_data=[data_test, data_test],
                            epochs=1,
                            shuffle=False,
                            batch_size=batch,
                            verbose=0,
                            initial_epoch=ep,
                            callbacks=callback)
            # print(story.history)
            # tot_loss_, reconstr_loss_, kl_loss_ = vae.train_on_batch(batchX)
            # print(dic)
            # print(float(story.history['loss'][0]))
            if len(story.history) != 0:
                tot_loss_ = float(story.history['loss'][0])
                reconstr_loss_ = float(story.history['reconstruction_loss'][0])
                kl_loss_ = float(story.history['kl_loss'][0])

            # print(tot_loss_, reconstr_loss_, kl_loss_)
            tot_loss += tot_loss_
            reconstr_loss += reconstr_loss_
            kl_loss += kl_loss_

            history['history']['total_loss'].append(tot_loss)
            history['history']['reconstruction_loss'].append(reconstr_loss)
            history['history']['kl_loss'].append(kl_loss)

        # save file h5
        if save:
            print("\t\t---->creating h5 file: " + savename + ".h5")
            vae.save(savename + ".h5", save_format='h5')

        # plot model
        # print(history)
        #
        epochs = range(epo)
        for key in history['history'].keys():
            if key != 'kl_loss':
                plt.plot(epochs, history['history'][key], "g--", color=next(colors), label=key)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"VAE Training Loss with {layer_num} layers")
        plt.legend()

        figure_num = 1
        while os.path.exists(f'{savename}_keras_loss_{figure_num}.png'):
            figure_num += 1
        plt.savefig(f'{savename}_keras_loss_{figure_num}.png')
        plt.clf()



    counter = 0
    fidelity = 0
    # save new files
    if savefiles:
        for file in wavefuns:
            name = f"{counter}_wavefun__{model.getInfo()}.txt"
            # predict probabilities
            #tmp = np.array(vae.decoder.predict([np.array([file])])[0])
            tmp = np.array(vae.predict([np.array([file])])[0])
            # print(tmp-file)

            # save to file
            fil = open(folderSaveNew + name, "wb")
            np.save(fil, tmp)
            fil.close()

            fidelity += theirFidelity(tmp, file)
            if counter == maximum:
                break
            counter += 1
    else:
        for i in np.random.randint(len(wavefuns), size=batch):
            file = wavefuns[i]

            # predict probabilities
            #tmp = np.array(vae.encoder.predict(np.array([file]))[0])
            #tmp = np.array(vae.decoder.predict([tmp])[0])
            tmp = np.array(vae.predict([np.array([file])])[0])
            # print(file, tmp, "\n\n\n")

            # fidelity np.dot(file, tmp)
            fidelity += theirFidelity(tmp, file)
            if counter == maximum:
                break
            counter += 1

    return vae, fidelity / counter  # ,entropy_before/counter, entropy/counter
