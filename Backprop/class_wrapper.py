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

# Libs
import numpy as np
from math import inf
import matplotlib.pyplot as plt
import pandas as pd
# Own module
from utils.time_recorder import time_keeper
from utils.helper_functions import simulator

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
        self.loss = self.make_loss()                            # The loss function
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

    def make_loss(self, logit=None, labels=None):
        """
        Create a tensor that represents the loss. This is consistant both at training time \
        and inference time for Backward model
        :param logit: The output of the network
        :param labels: The ground truth labels
        :return: the total loss
        """
        if logit is None:
            return None
        #if self.flags.data_set != 'gaussian_mixture':
        MSE_loss = nn.functional.mse_loss(logit, labels)          # The MSE Loss
        BDY_loss = 0
        self.MSE_loss = MSE_loss
        self.Boundary_loss = BDY_loss
        return torch.add(MSE_loss, BDY_loss)
        #else:                           # This is cross entropy loss where data is categorical
        #    criterion = nn.CrossEntropyLoss()
        #    return criterion(logit, labels.long())

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

        # Time keeping
        tk = time_keeper(time_keeping_file=os.path.join(self.ckpt_dir, 'training time.txt'))

        for epoch in range(self.flags.train_step):
            # Set to Training Mode
            train_loss = 0
            # boundary_loss = 0                 # Unnecessary during training since we provide geometries
            self.model.train()
            for j, (geometry, spectra) in enumerate(self.train_loader):
                if self.flags.data_set == 'gaussian_mixture':
                    spectra = torch.nn.functional.one_hot(spectra.to(torch.int64), 4).to(torch.float) # Change the gaussian labels into one-hot
                if cuda:
                    geometry = geometry.cuda()                          # Put data onto GPU
                    spectra = spectra.cuda()                            # Put data onto GPU
                self.optm.zero_grad()                               # Zero the gradient first
                logit = self.model(geometry)                        # Get the output
                loss = self.make_loss(logit, spectra)               # Get the loss tensor
                loss.backward()                                     # Calculate the backward gradients
                self.optm.step()                                    # Move one step the optimizer
                train_loss += loss                                  # Aggregate the loss
                # boundary_loss += self.Boundary_loss                 # Aggregate the BDY loss

            # Calculate the avg loss of training
            train_avg_loss = train_loss.cpu().data.numpy() / (j + 1)
            # boundary_avg_loss = boundary_loss.cpu().data.numpy() / (j + 1)

            if epoch % self.flags.eval_step:                      # For eval steps, do the evaluations and tensor board
                # Record the training loss to the tensorboard
                self.log.add_scalar('Loss/train', train_avg_loss, epoch)
                # self.log.add_scalar('Loss/BDY_train', boundary_avg_loss, epoch)

                # Set to Evaluation Mode
                self.model.eval()
                print("Doing Evaluation on the model now")
                test_loss = 0
                for j, (geometry, spectra) in enumerate(self.test_loader):  # Loop through the eval set
                    if self.flags.data_set == 'gaussian_mixture':
                        spectra = torch.nn.functional.one_hot(spectra.to(torch.int64), 4).to(torch.float) # Change the gaussian labels into one-hot
                    if cuda:
                        geometry = geometry.cuda()
                        spectra = spectra.cuda()
                    logit = self.model(geometry)
                    loss = self.make_loss(logit, spectra)                   # compute the loss
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
                        break

            # Learning rate decay upon plateau
            self.lr_scheduler.step(train_avg_loss)
        self.log.close()
        tk.record(1)                    # Record at the end of the training

    def evaluate(self, save_dir='data/', save_all=False):
        self.load()                             # load the model as constructed
        try:
            bs = self.flags.backprop_step         # for previous code that did not incorporate this
        except AttributeError:
            print("There is no attribute backprop_step, catched error and adding this now")
            self.flags.backprop_step = 2
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.model.cuda()
        self.model.eval()
        saved_model_str = self.saved_model.replace('/','_')
        # Get the file names
        Ypred_file = os.path.join(save_dir, 'test_Ypred_{}.csv'.format(saved_model_str))
        Xtruth_file = os.path.join(save_dir, 'test_Xtruth_{}.csv'.format(saved_model_str))
        Ytruth_file = os.path.join(save_dir, 'test_Ytruth_{}.csv'.format(saved_model_str))
        Xpred_file = os.path.join(save_dir, 'test_Xpred_{}.csv'.format(saved_model_str))
        print("evalution output pattern:", Ypred_file)

        # Time keeping
        tk = time_keeper(time_keeping_file=os.path.join(save_dir, 'evaluation_time.txt'))

        # Open those files to append
        with open(Xtruth_file, 'a') as fxt,open(Ytruth_file, 'a') as fyt,\
                open(Ypred_file, 'a') as fyp, open(Xpred_file, 'a') as fxp:
            # Loop through the eval data and evaluate
            for ind, (geometry, spectra) in enumerate(self.test_loader):
                if self.flags.data_set == 'gaussian_mixture':
                    spectra = torch.nn.functional.one_hot(spectra.to(torch.int64), 4).to(torch.float) # Change the gaussian labels into one-hot
                if cuda:
                    geometry = geometry.cuda()
                    spectra = spectra.cuda()
                # Initialize the geometry first
                Xpred, Ypred, loss = self.evaluate_one(spectra, save_dir=save_dir, save_all=save_all, ind=ind)
                tk.record(ind)                          # Keep the time after each evaluation for backprop
                # self.plot_histogram(loss, ind)                                # Debugging purposes
                np.savetxt(fxt, geometry.cpu().data.numpy(), fmt='%.3f')
                np.savetxt(fyt, spectra.cpu().data.numpy(), fmt='%.3f')
                np.savetxt(fyp, Ypred, fmt='%.3f')
                np.savetxt(fxp, Xpred, fmt='%.3f')
        return Ypred_file, Ytruth_file

    def evaluate_one(self, target_spectra, save_dir='data/', save_all=False, ind=None):
        print("evaluate_one gets save_dir:", save_dir)
        if torch.cuda.is_available():                                   # Initialize UNIFORM RANDOM NUMBER
            geometry_eval = torch.rand([self.flags.eval_batch_size, self.flags.linear[0]], requires_grad=True, device='cuda')
        else:
            geometry_eval = torch.rand([self.flags.eval_batch_size, self.flags.linear[0]], requires_grad=True) 
        
        self.optm_eval = self.make_optimizer_eval(geometry_eval)
        self.lr_scheduler = self.make_lr_scheduler(self.optm_eval)
        # expand the target spectra to eval batch size
        target_spectra_expand = target_spectra.expand([self.flags.eval_batch_size, -1])
        # Start backprop
        #print("shape of logit", np.shape(logit))
        #print("shape of target_spectra_expand", np.shape(target_spectra_expand))
        #print("shape of geometry_eval", np.shape(geometry_eval))
        Best_MSE_list = []
        Avg_MSE_list = []
        Xpred_best = None
        Best_MSE = 999
        save_all_Best_MSE_list = np.ones([self.flags.eval_batch_size, 1]) * 999
        save_all_Xpred_best = np.zeros_like(geometry_eval.cpu().data.numpy())
        save_all_Ypred_best = None
        # Define the full loss matrix, real means simulator loss, fake means NN loss
        Full_loss_matrix_real = np.zeros([self.flags.eval_batch_size, self.flags.backprop_step])
        Full_loss_matrix_fake = np.zeros([self.flags.eval_batch_size, self.flags.backprop_step])

        for i in range(self.flags.backprop_step):
            # Make the initialization from [-1, 1]
            geometry_eval_input = geometry_eval * 2 - 1
            self.optm_eval.zero_grad()                                  # Zero the gradient first
            logit = self.model(geometry_eval_input)                     # Get the output
            loss = self.make_loss(logit, target_spectra_expand)         # Get the loss
            loss.backward()                                             # Calculate the Gradient
            self.optm_eval.step()                                       # Move one step the optimizer
            
            ###################################
            # evaluate through simulator part #
            ###################################
            Ypred = simulator(self.flags.data_set, geometry_eval.cpu().data.numpy())
            if len(np.shape(Ypred)) == 1:                           # If this is the ballistics dataset where it only has 1d y'
                Ypred = np.reshape(Ypred, [-1, 1])
            # Get the MSE list of these
            MSE_list = np.mean(np.square(Ypred - target_spectra_expand.cpu().data.numpy()), axis=1)
            # Get the best and the index of it
            best_MSE_in_batch = np.min(MSE_list)
            avg_MSE_in_batch = np.mean(MSE_list)
            Best_MSE_list.append(best_MSE_in_batch)
            Avg_MSE_list.append(avg_MSE_in_batch)
            best_estimate_index = np.argmin(MSE_list)
            if best_MSE_in_batch < Best_MSE:
                # Update the best one
                Best_MSE = best_MSE_in_batch
                # Get the best Xpred
                Xpred_best = np.reshape(np.copy(geometry_eval_input.cpu().data.numpy()[best_estimate_index, :]), [1, -1])
                Ypred_best = np.reshape(np.copy(Ypred[best_estimate_index, :]), [1, -1])
            # If choose the record the process
            if save_all: 
                # In the first epoch this is none, assign value to this
                if save_all_Ypred_best is None:
                    save_all_Ypred_best = Ypred
                # Record the trails that gets better
                #print("shape of MSE_list", np.shape(MSE_list))
                #print("shape of save_all_best", np.shape(save_all_Best_MSE_list))
                MSE_list = np.reshape(MSE_list, [-1, 1])
                better_index = save_all_Best_MSE_list > MSE_list
                # Update those MSE that is better now
                save_all_Best_MSE_list = np.where(better_index, MSE_list, save_all_Best_MSE_list)
                save_all_Xpred_best = np.where(better_index, geometry_eval_input.cpu().data.numpy(), save_all_Xpred_best)
                save_all_Ypred_best = np.where(better_index, Ypred, save_all_Ypred_best)
                #print("shape of best MSE List", np.shape(save_all_Best_MSE_list))
                #print("shape of Xpred best", np.shape(save_all_Xpred_best))
                #print("shape of Ypred best", np.shape(save_all_Ypred_best))

            # record the full loss matrix
            Full_loss_matrix_real[:, i] = np.squeeze(MSE_list)
            Real_MSE_list = np.mean(np.square(logit.cpu().data.numpy() - target_spectra_expand.cpu().data.numpy()), axis=1)
            Full_loss_matrix_fake[:, i] = np.copy(Real_MSE_list)
            
            # Learning rate decay upon plateau
            self.lr_scheduler.step(loss.data)
            """
            ##########################################
            # Old version before change to simulator #
            ##########################################
            # check periodically to stop and print stuff
            if i % self.flags.eval_step == 0:
                print("loss at inference step{} : {}".format(i, loss.data))     # Print loss
                #print("printing the first 5 geometry_eval_input")
                #print(self.model.geometry_eval_input.cpu().data.numpy()[0:5,:])
                if loss.data < self.flags.stop_threshold:                       # Check if stop
                    print("Loss is lower than threshold{}, inference stop".format(self.flags.stop_threshold))
                    break
            """
        
        # Save the Best_MSE list for first few to sample
        if ind < 20:
            np.savetxt('best_mse/best_mse_list{}.csv'.format(ind), Best_MSE_list)
            np.savetxt('best_mse/avg_mse_list{}.csv'.format(ind), Avg_MSE_list)
            np.savetxt('best_mse/full_loss_mat_real{}.csv'.format(ind), Full_loss_matrix_real)
            np.savetxt('best_mse/full_loss_mat_fake{}.csv'.format(ind), Full_loss_matrix_fake)

        if save_all:
            for i in range(len(geometry_eval_input.cpu().data.numpy())):
                saved_model_str = self.saved_model.replace('/', '_') + 'inference' + str(i)
                Ypred_file = os.path.join(save_dir, 'test_Ypred_{}.csv'.format(saved_model_str))
                Xpred_file = os.path.join(save_dir, 'test_Xpred_{}.csv'.format(saved_model_str))
                # 2 options: simulator/logit
                #Ypred = simulator(self.flags.data_set, geometry_eval.cpu().data.numpy())
                #if len(np.shape(Ypred)) == 1:           # If this is the ballistics dataset where it only has 1d y'
                #    Ypred = np.reshape(Ypred, [-1, 1])
                #ypred = np.reshape(Ypred[i,:], [1, -1])
                ##ypred = np.reshape(logit.cpu().data.numpy()[i,:], [1, -1])
                #xpred = np.reshape(geometry_eval_input.cpu().data.numpy()[i,:], [1, -1])
                with open(Xpred_file, 'a') as fxp, open(Ypred_file, 'a') as fyp:
                    np.savetxt(fyp, save_all_Ypred_best[i, :], fmt='%.3f')
                    np.savetxt(fxp, save_all_Xpred_best[i, :], fmt='%.3f')
       
        """
        ###########################
        # Old version of Backprop #
        ###########################
        # Get the best performing one, 2 possibility, logit / simulator
        Ypred = simulator(self.flags.data_set, geometry_eval.cpu().data.numpy()) 
        if len(np.shape(Ypred)) == 1:           # If this is the ballistics dataset where it only has 1d y'
            Ypred = np.reshape(Ypred, [-1, 1])
        #MSE_list = np.mean(np.square(logit.cpu().data.numpy() - target_spectra_expand.cpu().data.numpy()), axis=1)
        MSE_list = np.mean(np.square(Ypred - target_spectra_expand.cpu().data.numpy()), axis=1)
        #print("shape of MSE list", np.shape(MSE_list))
        best_estimate_index = np.argmin(MSE_list)
        #print("best_estimate_index = ", best_estimate_index, " best error is ", MSE_list[best_estimate_index])
        Xpred_best = np.reshape(np.copy(geometry_eval_input.cpu().data.numpy()[best_estimate_index, :]), [1, -1])
        Ypred_best = np.reshape(np.copy(Ypred[best_estimate_index, :]), [1, -1])
        #print("the shape of Xpred_best is", np.shape(Xpred_best))
        """
        return Xpred_best, Ypred_best, MSE_list


    def predict(self, Xpred_file):
        """
        The prediction function, takes Xpred file and write Ypred file using trained model
        :param Xpred_file: Xpred file by (usually VAE) for meta-material
        :return: pred_file, truth_file to compare
        """
        self.load()         # load the model
        Ypred_file = Xpred_file.replace('Xpred', 'Ypred')
        Xpred = pd.read_csv(Xpred_file, header=None, delimiter=' ')     # Read the input
        Xpred.info()
        Xpred_tensor = torch.from_numpy(Xpred.values).to(torch.float)

        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.model.cuda()
            Xpred_tensor = Xpred_tensor.cuda()
        with open(Ypred_file, 'a') as fyp:
            Ypred = self.model(Xpred_tensor)
            np.savetxt(fyp, Ypred.cpu().data.numpy(), fmt='%.3f')

        Ytruth_file = Ypred_file.replace('Ypred', 'Ytruth')
        return Ypred_file, Ytruth_file



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
