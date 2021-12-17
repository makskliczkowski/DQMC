from src.common.__commonFuns__ import *

# PYTORCH
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset



torch.cuda.empty_cache()

class Print(nn.Module):
    """
    Used to print the shape of the data inside of nn.Sequential
    """

    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x


def get_layers(input_size, n_layers, compression):
    reduction = compression // n_layers
    encoder = None
    # These if statements determine the structure of the VAE dependent on the desired compression
    print(input_size, reduction)
    if n_layers == 1:
        # Desired structure for 1 total layer in the encoder
        encoder = nn.Sequential(nn.Sigmoid())
    elif n_layers == 2:
        # Desired structure for 2 total layers in the encoder
        encoder = nn.Sequential(
            nn.Linear(input_size, input_size - reduction),
            nn.Sigmoid()
        )
    elif n_layers == 3:
        # Desired structure for 3 total layers in the encoder
        encoder = nn.Sequential(
            nn.Linear(input_size, input_size - reduction),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction,
                      input_size - reduction * 2),
            nn.Sigmoid()
        )
    elif n_layers == 4:
        # Desired structure for 4 total layers in the encoder
        encoder = nn.Sequential(
            nn.Linear(input_size, input_size - reduction),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction,
                      input_size - reduction * 2),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 2,
                      input_size - reduction * 3),
            nn.Sigmoid()
        )
    elif n_layers == 5:
        # Desired structure for 5 total layers in the encoder
        encoder = nn.Sequential(
            nn.Linear(input_size, input_size - reduction),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction,
                      input_size - reduction * 2),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 2,
                      input_size - reduction * 3),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 3,
                      input_size - reduction * 4),
            nn.Sigmoid()
        )
    elif n_layers == 6:
        # Desired structure for 6 total layers in the encoder
        encoder = nn.Sequential(
            nn.Linear(input_size, input_size - reduction),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction,
                      input_size - reduction * 2),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 2,
                      input_size - reduction * 3),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 3,
                      input_size - reduction * 4),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 4,
                      input_size - reduction * 5),
            nn.Sigmoid()
        )
    elif n_layers == 7:
        # Desired structure for 7 total layers in the encoder
        encoder = nn.Sequential(
            nn.Linear(input_size, input_size - reduction),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction,
                      input_size - reduction * 2),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 2,
                      input_size - reduction * 3),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 3,
                      input_size - reduction * 4),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 4,
                      input_size - reduction * 5),
            nn.Linear(input_size - reduction * 5,
                      input_size - reduction * 6),
            nn.Sigmoid()
        )

    decoder = None

    if n_layers == 7:
        # Desired structure for 7 total layers in the decoder
        decoder = nn.Sequential(
            nn.Linear(compression, input_size - reduction * (n_layers - 1)),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 6,
                      input_size - reduction * 5),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 5,
                      input_size - reduction * 4),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 4,
                      input_size - reduction * 3),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 3,
                      input_size - reduction * 2),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 2,
                      input_size - reduction * 1),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 1, input_size),
            nn.Sigmoid()
        )
    elif (n_layers == 6):
        # Desired structure for 6 total layers in the decoder
        decoder = nn.Sequential(
            nn.Linear(compression, input_size - reduction * (n_layers - 1)),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 5,
                      input_size - reduction * 4),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 4,
                      input_size - reduction * 3),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 3,
                      input_size - reduction * 2),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 2,
                      input_size - reduction * 1),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 1, input_size),
            nn.Sigmoid()
        )
    elif (n_layers == 5):
        # Desired structure for 5 total layers in the decoder
        decoder = nn.Sequential(
            nn.Linear(compression, input_size - reduction * (n_layers - 1)),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 4,
                      input_size - reduction * 3),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 3,
                      input_size - reduction * 2),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 2,
                      input_size - reduction * 1),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 1, input_size),
            nn.Sigmoid()
        )
    elif n_layers == 4:
        # Desired structure for 4 total layers in the decoder
        decoder = nn.Sequential(
            nn.Linear(compression, input_size - reduction * (n_layers - 1)),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 3,
                      input_size - reduction * 2),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 2,
                      input_size - reduction * 1),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 1, input_size),
            nn.Sigmoid()
        )
    elif n_layers == 3:
        # Desired structure for 3 total layers in the decoder
        decoder = nn.Sequential(
            nn.Linear(compression, input_size - reduction * (n_layers - 1)),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 2,
                      input_size - reduction * 1),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 1, input_size),
            nn.Sigmoid()
        )
    elif n_layers == 2:
        # Desired structure for 2 total layers in the decoder
        decoder = nn.Sequential(
            nn.Linear(compression, input_size - reduction * (n_layers - 1)),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 1, input_size),
            nn.Sigmoid()
        )
    elif n_layers == 1:
        # Desired structure for 1 total layer in the decoder
        decoder = nn.Sequential(
            nn.Linear(input_size - reduction * 1, input_size),
            nn.Sigmoid()
        )

    # Latent log variance and mu layers
    fc_logvar = nn.Linear(input_size - reduction * (n_layers - 1), compression)
    fc_mu = nn.Linear(input_size - reduction * (n_layers - 1), compression)

    return {'decoder': decoder, 'encoder': encoder, 'logvar': fc_logvar, 'mu': fc_mu}


# Define your own class LoadFromFolder
class FromFolder(Dataset):
    def __init__(self, main_dir, filenum=None):
        # Set the loading directory
        self.main_dir = main_dir
        # List all images in folder and count them
        all_files = list(filter(lambda x: x.endswith('.dat'), os.listdir(self.main_dir)))
        if filenum != None:
            all_files = all_files[0:filenum]
        self.total_files = sorted(all_files)

    def __len__(self):
        # Return the previously computed number of images
        return len(self.total_files)

    def __getitem__(self, idx):
        loc = os.path.join(self.main_dir, self.total_files[idx])
        fil = np.square(np.genfromtxt(loc))
        return fil


class Modelik:
    def __init__(self, parameters, data_path, lat_dim, verbosity=0,
                 n_layers=3, n_qubits=8, trainsize=0.9, filenum=None,
                 early_stop = True, load=None):
        """
        Args:
            parameters: dict of json params
            n_layers: number of layers in the encoder/decoder
            n_qubits: number of lattice sites
            load: optional path to load a pretrained model
        """
        # Initialize class parameteres
        torch.cuda.empty_cache()

        self.n_layers = n_layers
        self.n_qubits = n_qubits
        self.N = int(math.pow(2, self.n_qubits))
        self.lat_dim = lat_dim
        self.compression = self.lat_dim / self.N

        self.epochs = int(parameters['epochs'])
        self.batch_size = int(parameters['batch_size'])
        self.trainsize = trainsize
        self.display_epochs = int(parameters['display_epoch'])
        self.learning_rate = parameters['learning_rate']
        self.num_batches = int(parameters['num_batches'])
        self.data_path = data_path
        self.filenum = filenum

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.verbosity = verbosity
        # early stoping
        self.early_stop = early_stop
        if self.early_stop:
            self.early_stopping = EarlyStopping(patience=10, min_delta=0.05)


        self.savename = self.data_path + f'..{kPSep}myModel_latent={self.lat_dim},{self.N}'

        # Prepare model
        self.vae, self.train_loaders, self.test_loaders, self.optimizer = self.prepare_model(
            load=load)

        # Train the model if it wasn't loaded, and compute fidelity
        if load == None:
            train_losses, test_losses = self.run_model()
            self.plot_losses(train_losses, test_losses)

        # print(self.train_loaders.dataset)
        self.fidelity = self.get_fidelity(self.train_loaders)

    def prepare_model(self, load=None):
        """
        Initializes VAE model and loads it onto the appropriate device.
        Reads and loads the data in the form of an array of Torch DataLoaders.
        Initializes Adam optimizer.
        Args:
            load: path to load trained model from
        Returns:
            VAE
            Array of train Torch Dataloaders
            Array of test Torch Dataloaders
            Adam optimizer
        Raises:
        """
        input_size = self.N
        VAE_layers = get_layers(input_size, self.n_layers, self.lat_dim)

        vae = VariationalAutoencoder(VAE_layers.get('encoder'), VAE_layers.get(
            'decoder'), VAE_layers.get('logvar'), VAE_layers.get('mu')).double().to(self.device)

        train_loaders, test_loaders = self.get_data(self.batch_size, self.data_path)

        optimizer = optim.Adam(vae.parameters(), lr=self.learning_rate)

        if not load == None:
            vae.load_state_dict(torch.load(load))
            vae.eval()

        return vae, train_loaders, test_loaders, optimizer

    def loss_function(self, x, x_reconstruction, mu, log_var, weight=1):
        """
        Returns the loss for the model based on the reconstruction likelihood and KL divergence
        Args:
            x: Input data
            x_reconstruction: Reconstructed data
            mu:
            log_var:
            weight:
        Returns:
            loss:
        Raises:
        """
        reconstruction_likelihood = F.binary_cross_entropy(
            x_reconstruction, x, reduction='sum')
        kl_divergence = -0.5 * \
                        torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        loss = reconstruction_likelihood + kl_divergence * weight

        return loss

    def get_fidelity(self, x):
        """
        Calculates the reconstruction fidelity.

        Args:
            x: Input data
        Returns:
            out: Fidelity for the input sample
        Raises:
        """
        # self.vae.eval()
        # torch.no_grad()
        fidel = 0
        counter = 0

        #x = x.dataset
        # x = x.dot(1 << np.arange(x.shape[-1] - 1, -1, -1))  # Converts binary string to integer
        #print(x)

        #l, u = x.min(), x.max() + 1
        #f1, b = np.histogram(x, density=True, bins=np.arange(l, u, 1))

        # Initialize for getting reconstructed density
        #f2 = np.zeros(f1.shape)
        #ns = 0
        #dim = int(self.n_qubits * self.compression)
        #while ns < 10:
            # Get samples, decode them, convert to int, and add to hist count
        #    re = np.random.multivariate_normal(
        #        np.zeros(dim), np.eye(dim), size=int(0.375e7))
        #    re = self.vae.decode(torch.Tensor(re).double().to(
        #        self.device)).cpu().detach().numpy()
        #    f2 += np.histogram(re, bins=b)[0]
        #    ns += 1

        with torch.no_grad():
            for i, data in enumerate(x):
                if i >= self.num_batches:
                    break

                #print(data)
                tmp = data.to(self.device)
                reconstruction_data, mu, logvar = self.vae(tmp)
                reconstruction_data = reconstruction_data.cpu()
                #print(reconstruction_data)
                for e in range(len(reconstruction_data)):
                    a = reconstruction_data[e]
                    b = data[e].numpy()
                    fidel += theirFidelity(a, b)
                    counter+=1
                    if counter >= self.batch_size:
                        break

        #db = np.array(np.diff(b), float)
        #f2 = f2 / db / f2.sum()

        #out = np.sum(np.sqrt(np.multiply(f1, f2)))
        #print(f"Fidelity: {out}")
        #del re, f1, f2, x
        del reconstruction_data, x

        torch.cuda.empty_cache()
        return fidel/counter

    def train(self, epoch, loader):
        """
        Trains the VAE model
        Args:
            epoch: Number of current epoch to print
            loader: Torch DataLoader for a quantum state
        Returns:
            epoch_loss: Loss for the epoch
        Raises:
        """
        self.vae.train()
        epoch_loss = 0

        for i, data in enumerate(loader):

            if i >= self.num_batches:
                break

            data = data.to(self.device)
            self.optimizer.zero_grad()
            reconstruction_data, mu, log_var = self.vae(data)

            loss = self.loss_function(
                data, reconstruction_data, mu, log_var, weight=0.85 * (epoch / self.epochs))
            loss.backward()
            epoch_loss += loss.item() / (data.size(0) * self.num_batches)
            self.optimizer.step()

            if (self.verbosity == 0 or (
                    self.verbosity == 1 and (epoch + 1) % self.display_epochs == 0)) and i % self.batch_size == 0:
                print("Done batch: " + str(i) +
                      "\tCurr Loss: " + str(epoch_loss))

        if self.verbosity == 0 or (self.verbosity == 1 and (epoch + 1) % self.display_epochs == 0):
            print('Epoch [{}/{}]'.format(epoch + 1, self.epochs) +
                  '\tLoss: {:.4f}'.format(epoch_loss)
                  )

        return epoch_loss

    def test(self, epoch, loader):
        """
        Tests VAE model
        Args:
            epoch: Number of current epoch to print
            loader: Torch DataLoader for a quantum state
        Returns:
            epoch_loss: Loss for the epoch
        Raises:
        """
        self.vae.eval()
        epoch_loss = 0

        with torch.no_grad():
            for i, data in enumerate(loader):

                if i >= self.num_batches:
                    break

                data = data.to(self.device)
                reconstruction_data, mu, logvar = self.vae(data)
                loss = self.loss_function(
                    data, reconstruction_data, mu, logvar)
                epoch_loss += loss.item() / (data.size(0) * self.num_batches)

        return epoch_loss

    def run_model(self):
        """
        Args:
            state: Quantum state the model will be trained on
        Returns:
            test and training loss
        Raises:
        """

        train_loader, test_loader = self.train_loaders, self.test_loaders
        train_losses, test_losses = [], []

        print("Beginning Training:")
        for e in range(0, self.epochs):
            train_loss = self.train(e, train_loader)
            test_loss = self.test(e, test_loader)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            if self.early_stop:
                self.early_stopping(test_loss)
                if self.early_stopping.early_stop:
                    break
        print(f"Final train loss: {train_losses[-1]}\tFinal test loss: {test_losses[-1]}")

        # torch.save(self.vae.state_dict(),self.savename)

        return train_losses, test_losses

    def plot_losses(self, train_losses, test_losses):
        """
        Args:
            train_losses: list of training losses from run_model
            test_losses: list of testing losses from run_model
        Returns:
        Raises:
        """
        savename = self.savename
        epochs = np.arange(0, len(train_losses), 1)
        plt.plot(epochs, train_losses, "g-", label="Training Loss")
        plt.plot(epochs, test_losses, "b-", label="Testing Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"VAE Training Loss with {self.n_layers} layers")
        plt.legend()
        plt.xlim(0, len(train_losses))

        figure_num = 1
        while os.path.exists(f'{savename}_loss_{figure_num}.png'):
            figure_num += 1
        plt.savefig(f'{savename}_loss_{figure_num}.png')
        plt.clf()
        print(f'{savename}_loss_{figure_num}.png')

    def plot_fidelities(self, fs):
        """
        Args:
            fs: A list of Fidelities from each model
        Returns:
        Raises:
        """

        savename = self.savename
        epochs = np.arange(1, len(fs) + 1, 1)
        plt.plot(epochs, fs, "b--o", label="Fidelity")
        plt.xlabel("Layers")
        plt.xticks(ticks=epochs)
        plt.ylabel("Fidelity")
        plt.title("VAE Fidelities")
        plt.xlim(epochs.min(), epochs.max())

        figure_num = 1
        while os.path.exists(f'{savename}_fidelity_{figure_num}.png'):
            figure_num += 1
        plt.savefig(f'{savename}_fidelity_{figure_num}.png')
        plt.clf()
        print(f'{savename}_fidelity_{figure_num}.png')

    def get_data(self, batch_size, file_path):
        """
        Args:
            batch_size: Size of batches
            file_path: Path of file location
        Returns:
            train_loaders: Array of Torch DataLoaders representing quantum states for training
            test_loaders: Array of Torch DataLoaders representing quantum states for testing
        Raises:
        """
        ds = FromFolder(file_path, self.filenum)
        num = ds.__len__()
        num_train = int(num * self.trainsize)
        train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(ds, (num_train, num - num_train, 0))

        train_loaders = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        test_loaders = torch.utils.data.DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=True)
        return train_loaders, test_loaders



class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder class.
    Architecture:
        - x Fully connected layers
        - Sigmoid activation function
        - LeakyReLU activation function with slope of -0.2
    """

    def __init__(self, encode, decode, logvar, mu):
        """
        Very standard VAE, all the heavy lifting done elsewhere
        Args:
            encode: encoder input from hidden_layers
            decode: decoder layers
            logvar: logvar layer
            mu:mu layer
        """

        super(VariationalAutoencoder, self).__init__()
        self.LReLU = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

        self.fc_logvar = logvar
        self.fc_mu = mu

        self.encode = encode
        self.decode = decode

    def encoder(self, x):
        x = self.encode(x)
        return self.fc_mu(x), self.fc_logvar(x)

    def decoder(self, x):
        x = self.decode(x)
        return x

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        x = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(x)
        return reconstruction, mu, logvar



class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True