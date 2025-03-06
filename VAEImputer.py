import numpy as np
from scipy import sparse

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, _check_feature_names_in
from sklearn.preprocessing import OneHotEncoder
import sklearn
import sklearn.impute
import math
import pandas as pd
from pandas.api.types import is_numeric_dtype
import sklearn.compose
import torch
from torch.utils.data import DataLoader, TensorDataset

class VAEImputer(BaseEstimator, TransformerMixin):

    def __init__(self, iterations=1000, batch_size=128, split_size=5, code_size=5, encoder_hidden_sizes=[128, 64], decoder_hidden_sizes=[128, 64],
                    temperature=None, p_miss = 0.2, learning_rate = 0.001, tolerance=0.001, random_state=None):
        
        self.batch_size = batch_size
        self.iterations = iterations
        self.split_size = split_size
        self.code_size = code_size
        self.encoder_hidden_sizes = encoder_hidden_sizes
        self.decoder_hidden_sizes = decoder_hidden_sizes
        self.test_loss_function = torch.nn.MSELoss()
        self.p_miss = p_miss
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.random_state=random_state
        torch.set_default_dtype(torch.float32)
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
            )
        torch.set_default_device(self.device)
        torch.set_default_dtype(torch.float32)
        torch.set_grad_enabled(True)
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

    class Encoder():
        def __init__(self, VAEImputer, input_size):
            super(VAEImputer.Encoder, self).__init__()
            self.E_W1 = torch.nn.init.xavier_normal_(torch.empty((VAEImputer.encoder_hidden_sizes[0], input_size), requires_grad=True, device=VAEImputer.device))    # Data + Hint as inputs
            self.E_b1 = torch.zeros((VAEImputer.encoder_hidden_sizes[0]),requires_grad=True, device=VAEImputer.device)

            self.E_W2 = torch.nn.init.xavier_normal_(torch.empty((VAEImputer.encoder_hidden_sizes[1], VAEImputer.encoder_hidden_sizes[0],),requires_grad=True, device=VAEImputer.device))
            self.E_b2 = torch.zeros((VAEImputer.encoder_hidden_sizes[1]),requires_grad=True, device=VAEImputer.device)

            self.E_W3 = torch.nn.init.xavier_normal_(torch.empty((VAEImputer.code_size, VAEImputer.encoder_hidden_sizes[1]),requires_grad=True, device=VAEImputer.device))
            self.E_b3 = torch.zeros((VAEImputer.code_size), requires_grad=True, device=VAEImputer.device)   
        
            self.E_W4 = torch.nn.init.xavier_normal_(torch.empty((VAEImputer.split_size, VAEImputer.code_size),requires_grad=True, device=VAEImputer.device))
            self.E_b4 = torch.zeros((VAEImputer.split_size), requires_grad=True, device=VAEImputer.device)   

            self.E_W5 = torch.nn.init.xavier_normal_(torch.empty((VAEImputer.split_size, VAEImputer.code_size),requires_grad=True, device=VAEImputer.device))
            self.E_b5 = torch.zeros((VAEImputer.split_size), requires_grad=True, device=VAEImputer.device)  

        def forward(self, x):
            l1  = torch.nn.functional.linear(input=x, weight=self.E_W1, bias=self.E_b1)
            out1 = torch.nn.functional.tanh(l1)
            l2 = torch.nn.functional.linear(input=out1, weight=self.E_W2, bias=self.E_b2)
            out2 = torch.nn.functional.tanh(l2)
            l3 = torch.nn.functional.linear(input=out2, weight=self.E_W3, bias=self.E_b3)
            out3 = torch.nn.functional.tanh(l3)
            mean = torch.nn.functional.linear(input=out3, weight=self.E_W4, bias=self.E_b4)
            log_var = torch.nn.functional.linear(input=out3, weight=self.E_W5, bias=self.E_b5)
            return mean, log_var

        def parameters(self):
            params = [self.E_W1, self.E_b1, self.E_W2, self.E_b2, self.E_W3, self.E_b3, self.E_W4, self.E_b4, self.E_W5, self.E_b5]
            return params
        
        def load_state(self, params):
            self.E_W1 = params[0]
            self.E_b1 = params[1]
            self.E_W2 = params[2]
            self.E_b2 = params[3]
            self.E_W3 = params[4]
            self.E_b3 = params[5]
            self.E_W4 = params[6]
            self.E_b4 = params[7]
            self.E_W5 = params[8]
            self.E_b5 = params[9]

    class Decoder():
        def __init__(self, VAEImputer, input_size):
            super(VAEImputer.Decoder, self).__init__()
            self.D_W1 = torch.nn.init.xavier_normal_(torch.empty((VAEImputer.decoder_hidden_sizes[0], VAEImputer.split_size), requires_grad=True, device=VAEImputer.device))    # Data + Hint as inputs
            self.D_b1 = torch.zeros((VAEImputer.decoder_hidden_sizes[0]),requires_grad=True, device=VAEImputer.device)

            self.D_W2 = torch.nn.init.xavier_normal_(torch.empty((VAEImputer.decoder_hidden_sizes[1], VAEImputer.decoder_hidden_sizes[0]),requires_grad=True, device=VAEImputer.device))
            self.D_b2 = torch.zeros((VAEImputer.decoder_hidden_sizes[1]),requires_grad=True, device=VAEImputer.device)

            self.D_W3 = torch.nn.init.xavier_normal_(torch.empty((input_size, VAEImputer.decoder_hidden_sizes[1]),requires_grad=True, device=VAEImputer.device))
            self.D_b3 = torch.zeros((input_size), requires_grad=True, device=VAEImputer.device)   

        def forward(self, x):
            l1  = torch.nn.functional.linear(input=x, weight=self.D_W1, bias=self.D_b1)
            out1 = torch.nn.functional.tanh(l1)
            l2 = torch.nn.functional.linear(input=out1, weight=self.D_W2, bias=self.D_b2)
            out2 = torch.nn.functional.tanh(l2)
            l3 = torch.nn.functional.linear(input=out2, weight=self.D_W3, bias=self.D_b3)
            x_hat = torch.nn.functional.sigmoid(l3)
            return x_hat

        def parameters(self):
            params = [self.D_W1, self.D_b1, self.D_W2, self.D_b2, self.D_W3, self.D_b3]
            return params
        
        def load_state(self, params):
            self.D_W1 = params[0]
            self.D_b1 = params[1]
            self.D_W2 = params[2]
            self.D_b2 = params[3]
            self.D_W3 = params[4]
            self.D_b3 = params[5]
            
    class VAE():
        def __init__(self, VAEImputer, input_size):
            super(VAEImputer.VAE, self).__init__()
            self.encoder = VAEImputer.Encoder(VAEImputer, input_size)
            self.decoder = VAEImputer.Decoder(VAEImputer, input_size)

        def forward(self, x):
            mu, log_var = self.encoder.forward(x)
            code = self.reparameterize(mu, log_var)
            reconstucted = self.decoder.forward(code)
            return code, reconstucted, mu, log_var

        def reparameterize(self, mu, log_var):
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
            
        def parameters(self):
            params = self.encoder.parameters() + self.decoder.parameters()
            return params
        
        def load_state(self, params):
            self.encoder.load_state(params[0:10]) 
            self.decoder.load_state(params[10:]) 

    def fit(self, X, y=None):
        #print('working')
        self.variable_sizes = [1]*X.shape[1] #list of 1s the same lenght as the features of X
        
        self.encoder_hidden_sizes = [int(math.floor(X.shape[1]/2)), int(math.floor(X.shape[1]*3/10))]
        self.decoder_hidden_sizes = [int(math.floor(X.shape[1]*3/10)), int(math.floor(X.shape[1]/2))]
        self.split_size =int(math.floor(X.shape[1]/5))
        self.code_size=int(math.floor(X.shape[1]/5))
        
        #print(self.encoder_hidden_sizes)

        features = torch.from_numpy(X.to_numpy()) #X features
        features = torch.nan_to_num(features)
        features = features.to(dtype=torch.float32)
        features = features.to(device=self.device)
        

        num_samples = len(features)
        variable_masks = []
        for variable_size in self.variable_sizes:
            variable_mask = (torch.zeros(num_samples, 1).uniform_(0.0, 1.0) > self.p_miss).float()
            if variable_size > 1:
                variable_mask = variable_mask.repeat(1, variable_size)
            variable_masks.append(variable_mask)
        mask = torch.cat(variable_masks, dim=1)

        temperature = self.temperature
        self.model = self.VAE(VAEImputer=self, input_size=features.shape[1])
        
        inverted_mask = 1 - mask
        observed = features * mask
        missing = torch.randn_like(features)
        noisy_features = observed + missing*inverted_mask

        if self.learning_rate is not None:
            missing = torch.autograd.Variable(missing, requires_grad=True)
            self.optim = torch.optim.Adam(self.model.parameters(), weight_decay=0, lr=self.learning_rate)

        #pbar = tqdm(range(self.iterations))
        for iterations in range(self.iterations):
            train_ds = torch.utils.data.TensorDataset(features.float(), mask.float(), noisy_features.float())
            losses = [np.inf]
            for f, m, n in torch.utils.data.DataLoader(train_ds, 
                                                       batch_size=self.batch_size,
                                                         shuffle=True, 
                                                         generator=torch.Generator(device=self.device)):
                loss = self.train_batch(f, m, n)
                temp_loss = losses[-1]
                
                if temp_loss - loss < self.tolerance:
                    break
                
                losses.append(loss)
            #pbar.set_postfix({'loss': min(losses)})
            '''
            if iterations % 100 == 0 :
                print(f'Epoch {iterations} loss: {loss:.4f}')
            '''

        self._VAE_params = self.model.parameters()
        #print(len(self._VAE_params))
        return self
    
    def train_batch(self, features, mask, noisy_features):
        self.optim.zero_grad()
        #print(features.shape)
        #print(noisy_features.shape)
        #noise = torch.autograd.Variable(torch.FloatTensor(len(noisy_features), self.p_miss).normal_())
        _, reconstructed, mu, log_var = self.model.forward(noisy_features)
        #print(reconstructed.shape)
        #print(reconstructed)
        # reconstruction of the non-missing values
        reconstruction_loss = self.masked_reconstruction_loss_function(reconstructed,
                                                                  features,
                                                                  mask,
                                                                  self.variable_sizes)
        missing_loss = self.masked_reconstruction_loss_function(reconstructed, features, 1-mask, self.variable_sizes)
        #print(reconstruction_loss)
        loss = torch.sqrt(self.test_loss_function((mask * features + (1.0 - mask) * reconstructed), features))
        
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        #print(kld_loss)
        observed_loss = reconstruction_loss + kld_loss
        #loss = loss.type(torch.float32)
        #print(loss)
        observed_loss.backward()

        self.optim.step()

        return observed_loss.cpu().detach().numpy()

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def transform(self, X, y=None):
        self.model.load_state(self._VAE_params)
        self.variable_sizes = [1]*X.shape[1] #list of 1s the same lenght as the features of X
        features = torch.from_numpy(X.to_numpy()) #X features
        features = torch.nan_to_num(features)
        mask = torch.from_numpy(1-np.isnan(X.to_numpy()))
        inverted_mask = ~mask
        num_samples = len(features)
        observed = features * mask
        missing = torch.randn_like(features)
        noisy_features = observed + missing*inverted_mask
        
        f = features.to(dtype=torch.float32)
        f = f.to(device=self.device)
        
        m = mask.to(dtype=torch.float32)
        m = m.to(device=self.device)
        #print(m)
        n = noisy_features.to(dtype=torch.float32)
        n = n.to(device=self.device)
        #print(n)
        with torch.no_grad():
            _, reconstructed, _, _ = self.model.forward(n)
            #print(reconstructed)
            imputed = m*n + (1.0 - m)*reconstructed
        return imputed.cpu().numpy()

    def reconstruction_loss_function(self, reconstructed, original, variable_sizes, reduction="mean"):
        # by default use loss for binary variables
        if variable_sizes is None:
            return torch.nn.functional.binary_cross_entropy(reconstructed, original, reduction=reduction)
        # use the variable sizes when available
        else:
            loss = 0
            start = 0
            numerical_size = 0
            for variable_size in variable_sizes:
                # if it is a categorical variable
                if variable_size > 1:
                    # add loss from the accumulated continuous variables
                    if numerical_size > 0:
                        end = start + numerical_size
                        batch_reconstructed_variable = reconstructed[:, start:end]
                        batch_target = original[:, start:end]
                        loss += torch.nn.functional.mse_loss(batch_reconstructed_variable, batch_target, reduction=reduction)
                        start = end
                        numerical_size = 0
                    # add loss from categorical variable
                    end = start + variable_size
                    batch_reconstructed_variable = reconstructed[:, start:end]
                    batch_target = torch.argmax(original[:, start:end], dim=1)
                    loss += torch.nn.functional.cross_entropy(batch_reconstructed_variable, batch_target, reduction=reduction)
                    start = end
                # if not, accumulate numerical variables
                else:
                    numerical_size += 1

            # add loss from the remaining accumulated numerical variables
            if numerical_size > 0:
                end = start + numerical_size
                batch_reconstructed_variable = reconstructed[:, start:end]
                batch_target = original[:, start:end]
                loss += torch.nn.functional.mse_loss(batch_reconstructed_variable, batch_target, reduction=reduction)

            return loss

    def masked_reconstruction_loss_function(self, reconstructed, original, mask, variable_sizes):
        return self.reconstruction_loss_function(mask * reconstructed,
                                            mask * original,
                                            variable_sizes,
                                            reduction="sum") / torch.sum(mask)