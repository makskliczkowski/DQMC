from src.models.__ising__ import *
from src.ml.__mlKeras__ import *
from src.ml.__mlTorch__ import *

destination = f"..{kPSep}resultsWAVE5{kPSep}disorder{kPSep}PBC{kPSep}"
L = 12
g = 0.8
h = 0.83
J = 1.0
w = 0.1
modelik = IsingDisorder(L, J, 0, g, 0, h, w, 0)
N = modelik.N

latent_dims = np.linspace(5, N, 5)
ws = np.array([0.1 * i for i in range(1, 30)])
epo = 20
batch = 30
filenum = 600
layer_num = 2

# for pytorch mode
parameters = {"epochs": epo, 'batch_size': batch, 'display_epoch': 5, 'learning_rate': 1e-3, 'num_batches': batch}
# the types of autoencoder
autoencoderTypes = {0: "keras", 1: "torch"}

#print(tf.version.VERSION)
def main():
    for layer_num in range(2, 4):
        with open(destination + f"heatmap_layernum={layer_num},filenum={filenum},L={L},epo={epo},batch={batch}.txt", "w") as mapka:
            for w in ws:
                modelik = IsingDisorder(L, J, 0, g, 0, h, w, 0)
                N = modelik.N
                folderLog = destination + "_" + modelik.getInfo() + kPSep
                folder = folderLog + "wavefunctions" + kPSep

                for lat_dim in latent_dims:
                    #modelik2 = Modelik(parameters, folder, lat_dim=int(lat_dim), verbosity=0,
                    #                   n_layers=layer_num,
                    #                   n_qubits=L, filenum=filenum,
                    #                   load=None)
                    #fidelity = modelik2.fidelity
                    enc, fidelity = fileAutoencode(destination, modelik, int(lat_dim), epo,
                                                   layer_num=layer_num,
                                                   filenum=filenum, batch=batch,
                                                   verbose=0,trainAll=True,
                                                   save=False, savefiles=False)

                    justPrinter(mapka
                                , "\t"
                                , [f"{w:.2f}"
                                    , f"{int(lat_dim) / float(N):.5f}"
                                    , f"{fidelity:.7f}"]
                                , width=10)
                    mapka.flush()

# doing stuff
main()
