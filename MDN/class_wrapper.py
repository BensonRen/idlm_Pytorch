"""
The class wrapper for the networks
"""
# Built-in
import os
import time
import sys
sys.path.append('../utils/')

# Torch
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
# from torchsummary import summary
from torch.optim import lr_scheduler
import mdn
# Libs
import numpy as np
from math import inf
import matplotlib.pyplot as plt
# Own module


class Network(object):
    def __init__(self, model_fn, flags, train_loader, test_loader,
                 ckpt_dir=os.path.join(os.path.abspath(''), 'models'),
                 inference_mode=False, saved_model=None):
        self.model_fn = model_fn                                # The model maker function
        self.flags = flags                                      # The Flags containing the specs
        if inference_mode:                                      # If inference mode, use saved model
            self.ckpt_dir = os.path.join(ckpt_dir, saved_model)
            self.saved_model = saved_model
            print("This is inference mode, the ckpt is", self.ckpt_dir)
        else:                                                   # training mode, create a new ckpt folder
            if flags.model_name is None:                    # leave custume name if possible
                self.ckpt_dir = os.path.join(ckpt_dir, time.strftime('%Y%m%d_%H%M%S', time.localtime()))
            else:
                self.ckpt_dir = os.path.join(ckpt_dir, flags.model_name)
        self.model = self.create_model()                        # The model itself
        self.optm = None                                        # The optimizer: Initialized at train() due to GPU
        self.optm_eval = None                                   # The eval_optimizer: Initialized at eva() due to GPU
        self.lr_scheduler = None                                # The lr scheduler: Initialized at train() due to GPU
        self.train_loader = train_loader                        # The train data loader
        self.test_loader = test_loader                          # The test data loader
        self.log = SummaryWriter(self.ckpt_dir)     # Create a summary writer for keeping the summary to the tensor board
        self.best_validation_loss = float('inf')    # Set the BVL to large number

    def make_optimizer_eval(self, geometry_eval):
        """
        The function to make the optimizer during evaluation time.
        The difference between optm is that it does not have regularization and it only optmize the self.geometr_eval tensor
        :return: the optimizer_eval
        """
        if self.flags.optim == 'Adam':
            op = torch.optim.Adam([geometry_eval], lr=self.flags.lr)
        elif self.flags.optim == 'RMSprop':
            op = torch.optim.RMSprop([geometry_eval], lr=self.flags.lr)
        elif self.flags.optim == 'SGD':
            op = torch.optim.SGD([geometry_eval], lr=self.flags.lr)
        else:
            raise Exception("Your Optimizer is neither Adam, RMSprop or SGD, please change in param or contact Ben")
        return op

    def create_model(self):
        """
        Function to create the network module from provided model fn and flags
        :return: the created nn module
        """
        model = self.model_fn(self.flags)
        # summary(model, input_size=(128, 8))
        print(model)
        return model

    def make_loss(self, pi, sigma, mu, labels=None):
        """
        The special loss for mdn
        :param logit: The output of the network
        :param labels: The ground truth labels
        :return: the total loss
        """
        return mdn.mdn_loss(pi, sigma, mu, labels)

    def make_optimizer(self):
        """
        Make the corresponding optimizer from the flags. Only below optimizers are allowed. Welcome to add more
        :return:
        """
        if self.flags.optim == 'Adam':
            op = torch.optim.Adam(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif self.flags.optim == 'RMSprop':
            op = torch.optim.RMSprop(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif self.flags.optim == 'SGD':
            op = torch.optim.SGD(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        else:
            raise Exception("Your Optimizer is neither Adam, RMSprop or SGD, please change in param or contact Ben")
        return op

    def make_lr_scheduler(self, optm):
        """
        Make the learning rate scheduler as instructed. More modes can be added to this, current supported ones:
        1. ReduceLROnPlateau (decrease lr when validation error stops improving
        :return:
        """
        return lr_scheduler.ReduceLROnPlateau(optimizer=optm, mode='min',
                                              factor=self.flags.lr_decay_rate,
                                              patience=10, verbose=True, threshold=1e-4)

    def save(self):
        """
        Saving the model to the current check point folder with name best_model_forward.pt
        :return: None
        """
        # torch.save(self.model.state_dict, os.path.join(self.ckpt_dir, 'best_model_state_dict.pt'))
        torch.save(self.model, os.path.join(self.ckpt_dir, 'best_model_forward.pt'))

    def load(self):
        """
        Loading the model from the check point folder with name best_model_forward.pt
        :return:
        """
        # self.model.load_state_dict(torch.load(os.path.join(self.ckpt_dir, 'best_model_state_dict.pt')))
        if torch.cuda.is_available():
            self.model = torch.load(os.path.join(self.ckpt_dir, 'best_model_forward.pt'))
        else:
            self.model = torch.load(os.path.join(self.ckpt_dir, 'best_model_forward.pt'), map_location=torch.device('cpu'))

    def train(self):
        """
        The major training function. This would start the training using information given in the flags
        :return: None
        """
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.model.cuda()

        # Construct optimizer after the model moved to GPU
        self.optm = self.make_optimizer()
        self.lr_scheduler = self.make_lr_scheduler(self.optm)

        for epoch in range(self.flags.train_step):
            # Set to Training Mode
            train_loss = 0
            # boundary_loss = 0                 # Unnecessary during training since we provide geometries
            self.model.train()
            for j, (geometry, spectra) in enumerate(self.train_loader):
                if cuda:
                    geometry = geometry.cuda()                          # Put data onto GPU
                    spectra = spectra.cuda()                            # Put data onto GPU
                self.optm.zero_grad()                               # Zero the gradient first
                pi, simga, mu = self.model(spectra)                        # Get the output
                loss = self.make_loss(pi, simga, mu, geometry)               # Get the loss tensor
                print('loss at epoch {}, batch {} is {}'.format(epoch, j, loss.detach().cpu().numpy()))
                loss.backward()                                     # Calculate the backward gradients
                self.optm.step()                                    # Move one step the optimizer
                train_loss += loss                                  # Aggregate the loss
                # boundary_loss += self.Boundary_loss                 # Aggregate the BDY loss

            # Calculate the avg loss of training
            train_avg_loss = train_loss.cpu().data.numpy() / (j + 1)
            # boundary_avg_loss = boundary_loss.cpu().data.numpy() / (j + 1)

            if epoch % self.flags.eval_step and False:                      # For eval steps, do the evaluations and tensor board
                # Record the training loss to the tensorboard
                self.log.add_scalar('Loss/train', train_avg_loss, epoch)
                # self.log.add_scalar('Loss/BDY_train', boundary_avg_loss, epoch)

                # Set to Evaluation Mode
                self.model.eval()
                print("Doing Evaluation on the model now")
                test_loss = 0
                for j, (geometry, spectra) in enumerate(self.test_loader):  # Loop through the eval set
                    if cuda:
                        geometry = geometry.cuda()
                        spectra = spectra.cuda()
                    pi, simga, mu = self.model(spectra)  # Get the output
                    loss = self.make_loss(pi, simga, mu, geometry)  # Get the loss tensor
                    test_loss += loss                                       # Aggregate the loss

                # Record the testing loss to the tensorboard
                test_avg_loss = test_loss.cpu().data.numpy() / (j+1)
                self.log.add_scalar('Loss/test', test_avg_loss, epoch)

                print("This is Epoch %d, training loss %.5f, validation loss %.5f" \
                      % (epoch, train_avg_loss, test_avg_loss ))
                # Plotting the first spectra prediction for validation
                # f = self.compare_spectra(Ypred=logit[0,:].cpu().data.numpy(), Ytruth=spectra[0,:].cpu().data.numpy())
                # self.log.add_figure(tag='spectra compare',figure=f,global_step=epoch)

                # Model improving, save the model down
                if test_avg_loss < self.best_validation_loss:
                    self.best_validation_loss = test_avg_loss
                    self.save()
                    print("Saving the model down...")

                    if self.best_validation_loss < self.flags.stop_threshold:
                        print("Training finished EARLIER at epoch %d, reaching loss of %.5f" %\
                              (epoch, self.best_validation_loss))
                        return None

            # Learning rate decay upon plateau
            self.lr_scheduler.step(train_avg_loss)
        self.log.close()

    def evaluate(self, save_dir='data/'):
        self.load()                             # load the model as constructed
        try:
            bs = self.flags.backprop_step         # for previous code that did not incorporate this
        except AttributeError:
            print("There is no attribute backprop_step, catched error and adding this now")
            self.flags.backprop_step = 500
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.model.cuda()
        self.model.eval()
        # Get the file names
        Ypred_file = os.path.join(save_dir, 'test_Ypred_{}.csv'.format(self.saved_model))
        Xtruth_file = os.path.join(save_dir, 'test_Xtruth_{}.csv'.format(self.saved_model))
        Ytruth_file = os.path.join(save_dir, 'test_Ytruth_{}.csv'.format(self.saved_model))
        Xpred_file = os.path.join(save_dir, 'test_Xpred_{}.csv'.format(self.saved_model))

        # Open those files to append
        with open(Xtruth_file, 'a') as fxt,open(Ytruth_file, 'a') as fyt,\
                open(Ypred_file, 'a') as fyp, open(Xpred_file, 'a') as fxp:
            # Loop through the eval data and evaluate
            for ind, (geometry, spectra) in enumerate(self.test_loader):
                if cuda:
                    geometry = geometry.cuda()
                    spectra = spectra.cuda()
                # Initialize the geometry first
                Xpred, Ypred, loss = self.evaluate_one(spectra)
                # self.plot_histogram(loss, ind)                                # Debugging purposes
                np.savetxt(fxt, geometry.cpu().data.numpy(), fmt='%.3f')
                np.savetxt(fyt, spectra.cpu().data.numpy(), fmt='%.3f')
                np.savetxt(fyp, Ypred, fmt='%.3f')
                np.savetxt(fxp, Xpred, fmt='%.3f')
        return Ypred_file, Ytruth_file

    def evaluate_one(self, target_spectra):
        if torch.cuda.is_available():
            geometry_eval = torch.randn([self.flags.eval_batch_size, self.flags.linear[0]], requires_grad=True, device='cuda')
        else:
            geometry_eval = torch.randn([self.flags.eval_batch_size, self.flags.linear[0]], requires_grad=True)
        self.optm_eval = self.make_optimizer_eval(geometry_eval)
        self.lr_scheduler = self.make_lr_scheduler(self.optm_eval)
        # expand the target spectra to eval batch size
        target_spectra_expand = target_spectra.expand([self.flags.eval_batch_size, -1])
        # Start backprop
        #print("shape of logit", np.shape(logit))
        #print("shape of target_spectra_expand", np.shape(target_spectra_expand))
        #print("shape of geometry_eval", np.shape(geometry_eval))
        for i in range(self.flags.backprop_step):
            logit = self.model(geometry_eval)                      # Get the output
            loss = self.make_loss(logit, target_spectra_expand)         # Get the loss
            loss.backward()                           # Calculate the Gradient
            self.optm_eval.step()                                       # Move one step the optimizer

            # check periodically to stop and print stuff
            if i % self.flags.eval_step == 0:
                print("loss at inference step{} : {}".format(i, loss.data))     # Print loss
                #print("printing the first 5 geometry_eval")
                #print(self.model.geometry_eval.cpu().data.numpy()[0:5,:])
                if loss.data < self.flags.stop_threshold:                       # Check if stop
                    print("Loss is lower than threshold{}, inference stop".format(self.flags.stop_threshold))
                    break

            # Learning rate decay upon plateau
            self.lr_scheduler.step(loss.data)

        # Get the best performing one
        MSE_list = np.mean(np.square(logit.cpu().data.numpy() - target_spectra_expand.cpu().data.numpy()), axis=1)
        #print("shape of MSE list", np.shape(MSE_list))
        best_estimate_index = np.argmin(MSE_list)
        #print("best_estimate_index = ", best_estimate_index, " best error is ", MSE_list[best_estimate_index])
        Xpred_best = np.reshape(np.copy(geometry_eval.cpu().data.numpy()[best_estimate_index, :]), [1, -1])
        Ypred_best = np.reshape(np.copy(logit.cpu().data.numpy()[best_estimate_index, :]), [1, -1])
        #print("the shape of Xpred_best is", np.shape(Xpred_best))

        return Xpred_best, Ypred_best, MSE_list
    

    def plot_histogram(self, loss, ind):
        """
        Plot the loss histogram to see the loss distribution
        """
        f = plt.figure()
        plt.hist(loss, bins=100)
        plt.xlabel('MSE loss')
        plt.ylabel('cnt')
        plt.suptitle('(Avg MSE={:4e})'.format(np.mean(loss)))
        plt.savefig(os.path.join('data','loss{}.png'.format(ind)))
        return None
        
    def compare_spectra(self, Ypred, Ytruth, T=None, title=None, figsize=[15, 5],
                        T_num=10, E1=None, E2=None, N=None, K=None, eps_inf=None):
        """
        Function to plot the comparison for predicted spectra and truth spectra
        :param Ypred:  Predicted spectra, this should be a list of number of dimension 300, numpy
        :param Ytruth:  Truth spectra, this should be a list of number of dimension 300, numpy
        :param title: The title of the plot, usually it comes with the time
        :param figsize: The figure size of the plot
        :return: The identifier of the figure
        """
        # Make the frequency into real frequency in THz
        fre_low = 0.8
        fre_high = 1.5
        frequency = fre_low + (fre_high - fre_low) / len(Ytruth) * np.arange(300)
        f = plt.figure(figsize=figsize)
        plt.plot(frequency, Ypred, label='Pred')
        plt.plot(frequency, Ytruth, label='Truth')
        if T is not None:
            plt.plot(frequency, T, linewidth=1, linestyle='--')
        if E2 is not None:
            for i in range(np.shape(E2)[0]):
                plt.plot(frequency, E2[i, :], linewidth=1, linestyle=':', label="E2" + str(i))
        if E1 is not None:
            for i in range(np.shape(E1)[0]):
                plt.plot(frequency, E1[i, :], linewidth=1, linestyle='-', label="E1" + str(i))
        if N is not None:
            plt.plot(frequency, N, linewidth=1, linestyle=':', label="N")
        if K is not None:
            plt.plot(frequency, K, linewidth=1, linestyle='-', label="K")
        if eps_inf is not None:
            plt.plot(frequency, np.ones(np.shape(frequency)) * eps_inf, label="eps_inf")
        # plt.ylim([0, 1])
        plt.legend()
        #plt.xlim([fre_low, fre_high])
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Transmittance")
        if title is not None:
            plt.title(title)
        return f
