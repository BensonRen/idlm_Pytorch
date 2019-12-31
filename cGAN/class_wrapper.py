"""
The class wrapper for the networks
"""
# Built-in
import os
import time

# Torch
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
# from torchsummary import summary
from torch.optim import lr_scheduler

# Libs
import numpy as np
from math import inf


# Own module


class Network(object):
    def __init__(self, model_fn_d, model_fn_g, model_fn_se, model_fn_f,
                 flags, train_loader, test_loader,
                 ckpt_dir=os.path.join(os.path.abspath(''), 'models'),
                 inference_mode=False, saved_model=None):
        """
        The initializer of the Network class which is the wrapper of our neural network, this is for the Tandem model
        :param model_fn_d: Model maker function for discriminator
        :param model_fn_g: Model maker function for generator
        :param model_fn_se: Model maker function for spectra encoder
        :param model_fn_f: Model maker function for forward model
        :param flags: The input param flag object
        :param train_loader: Train loader of data
        :param test_loader: Test data loader
        :param ckpt_dir: The directory to save the check point file, default is 'models'
        :param inference_mode: The boolean flag whether this is a inference mode
        :param saved_model: Whether to load from the saved model, if not None then load from that model
        """
        self.model_fn_d = model_fn_d  # The model maker function for discriminator
        self.model_fn_g = model_fn_g  # The model maker function for generator
        self.model_fn_se = model_fn_se  # The model maker function for spectra encoder
        self.model_fn_f = model_fn_f  # # The model maker function for forward
        self.flags = flags  # The Flags containing the specs
        if inference_mode:  # If inference mode, use saved model
            self.ckpt_dir = os.path.join(ckpt_dir, saved_model)
            self.saved_model = saved_model
            print("This is inference mode, the ckpt is", self.ckpt_dir)
        else:  # training mode, create a new ckpt folder
            self.ckpt_dir = os.path.join(ckpt_dir, time.strftime('%Y%m%d_%H%M%S', time.localtime()))
        self.model_d, self.model_g, self.model_f, self.model_se = self.create_model()  # The model itself
        self.loss = self.make_loss()  # The loss function
        self.optm_d = None  # The optimizer: Initialized at train() due to GPU
        self.optm_g = None  # The eval_optimizer: Initialized at eva() due to GPU
        self.optm_se = None  # The optimizer: Initialized at train() due to GPU
        self.optm_f = None  # The eval_optimizer: Initialized at eva() due to GPU
        self.lr_scheduler = None  # The lr scheduler: Initialized at train() due to GPU
        self.train_loader = train_loader  # The train data loader
        self.test_loader = test_loader  # The test data loader
        self.log = SummaryWriter(self.ckpt_dir)  # Create a summary writer for keeping the summary to the tensor board
        self.best_validation_loss = float('inf')  # Set the BVL to large number

    def make_optimizer_g(self):
        """
        The function to make the optimizer during evaluation time.
        The difference between optm is that it does not have regularization and it only optmize the self.geometr_eval tensor
        :return: the optimizer_eval
        """
        if self.flags.optim == 'Adam':
            op = torch.optim.Adam([*self.model_g.parameters(), *self.model_se.parameters()],
                                  lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif self.flags.optim == 'RMSprop':
            op = torch.optim.RMSprop([*self.model_g.parameters(), *self.model_se.parameters()],
                                     lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif self.flags.optim == 'SGD':
            op = torch.optim.SGD([*self.model_g.parameters(), *self.model_se.parameters()],
                                 lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        else:
            raise Exception("Your Optimizer is neither Adam, RMSprop or SGD, please change in param or contact Ben")
        return op

    def create_model(self):
        """
        Function to create the network module from provided model fn and flags
        :return: the created nn module
        """
        model_d = self.model_fn_d(self.flags)
        model_g = self.model_fn_g(self.flags)
        print("Discriminator model", model_d)
        print("Generator model", model_g)
        model_f = self.model_fn_f(self.flags)
        model_se = self.model_fn_se(self.flags)
        print("Forward model", model_f)
        print("Sectral_encoder model", model_se)
        return model_d, model_g, model_f, model_se

    def make_loss(self, logit=None, labels=None, G=None):
        """
        Create a tensor that represents the loss. This is consistant both at training time \
        and inference time for Backward model
        :param logit: The output of the network
        :param labels: The ground truth labels
        :param G: The geometry predicted
        :return: the total loss
        """
        if logit is None:
            return None
        MSE_loss = nn.functional.mse_loss(logit, labels)  # The MSE Loss
        BDY_loss = torch.zeros(size=[])
        # Boundary loss of the geometry_eval to be less than 1
        if G is not None:
            relu = torch.nn.ReLU()
            BDY_loss_all = relu(torch.abs(G) - 1)
            BDY_loss = torch.mean(BDY_loss_all)
        self.MSE_loss = MSE_loss
        self.Boundary_loss = BDY_loss
        return torch.add(MSE_loss, BDY_loss)

    def make_optimizer_d(self):
        """
        Make the corresponding optimizer from the flags. Only below optimizers are allowed. Welcome to add more
        :return:
        """
        if self.flags.optim == 'Adam':
            op = torch.optim.Adam([*self.model_d.parameters(), *self.model_se.parameters()],
                                  lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif self.flags.optim == 'RMSprop':
            op = torch.optim.RMSprop([*self.model_d.parameters(), *self.model_se.parameters()],
                                     lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif self.flags.optim == 'SGD':
            op = torch.optim.SGD([*self.model_d.parameters(), *self.model_se.parameters()],
                                 lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        else:
            raise Exception("Your Optimizer is neither Adam, RMSprop or SGD, please change in param or contact Ben")
        return op

    def make_optimizer_f(self):
        """
        Make the corresponding optimizer from the flags. Only below optimizers are allowed. Welcome to add more
        :return:
        """
        if self.flags.optim == 'Adam':
            op = torch.optim.Adam(self.model_f.parameters(),
                                  lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif self.flags.optim == 'RMSprop':
            op = torch.optim.RMSprop(self.model_f.parameters(),
                                     lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif self.flags.optim == 'SGD':
            op = torch.optim.SGD(self.model_f.parameters(),
                                 lr=self.flags.lr, weight_decay=self.flags.reg_scale)
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

    def save_d(self):
        """
        Saving the model to the current check point folder with name best_model.pt
        :return: None
        """
        # torch.save(self.model.state_dict, os.path.join(self.ckpt_dir, 'best_model_state_dict.pt'))
        torch.save(self.model_d, os.path.join(self.ckpt_dir, 'best_model_discriminator.pt'))

    def save_g(self):
        """
        Saving the model to the current check point folder with name best_model.pt
        :return: None
        """
        torch.save(self.model_g, os.path.join(self.ckpt_dir, 'best_model_generator.pt'))

    def save_f(self):
        """
        Saving the model to the current check point folder with name best_model.pt
        :return: None
        """
        torch.save(self.model_f, os.path.join(self.ckpt_dir, 'best_model_forward.pt'))

    def save_se(self):
        """
        Saving the model to the current check point folder with name best_model.pt
        :return: None
        """
        torch.save(self.model_se, os.path.join(self.ckpt_dir, 'best_model_spectra_encoder.pt'))

    def load(self):
        """
        Loading the model from the check point folder with name best_model.pt
        :return:
        """
        # self.model.load_state_dict(torch.load(os.path.join(self.ckpt_dir, 'best_model_state_dict.pt')))
        self.model_d = torch.load(os.path.join(self.ckpt_dir, 'best_model_discriminator.pt'))
        self.model_g = torch.load(os.path.join(self.ckpt_dir, 'best_model_generator.pt'))
        self.model_f = torch.load(os.path.join(self.ckpt_dir, 'best_model_forward.pt'))
        self.model_se = torch.load(os.path.join(self.ckpt_dir, 'best_model_spectra_encoder.pt'))

    def train_forward(self):
        """
        The training block for training the forward module. This is served as the basis of the model since it provides
        the metric for discriminator that whether this is a "good" example or "bad" example.
        This function is before you call the train funciton
        :return:
        """
        print("Start Forward Training now")
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.model_f.cuda()

        # Construct optimizer after the model moved to GPU
        self.optm_f = self.make_optimizer_f()
        self.lr_scheduler = self.make_lr_scheduler(self.optm_f)

        for epoch in range(self.flags.train_step):
            # print("This is training Epoch {}".format(epoch))
            # Set to Training Mode
            train_loss = 0
            self.model_f.train()
            for j, (geometry, spectra) in enumerate(self.train_loader):
                if cuda:
                    geometry = geometry.cuda()  # Put data onto GPU
                    spectra = spectra.cuda()  # Put data onto GPU
                self.optm_f.zero_grad()  # Zero the gradient first
                logit = self.model_f(geometry)  # Get the output
                loss = self.make_loss(logit, spectra)  # Get the loss tensor
                loss.backward()  # Calculate the backward gradients
                self.optm_f.step()  # Move one step the optimizer
                train_loss += loss  # Aggregate the loss

            # Calculate the avg loss of training
            train_avg_loss = train_loss.cpu().data.numpy() / (j + 1)

            if epoch % self.flags.eval_step == 0:  # For eval steps, do the evaluations and tensor board
                # Set to Evaluation Mode
                self.model_f.eval()
                print("Doing Evaluation on the model now")
                test_loss = 0
                for j, (geometry, spectra) in enumerate(self.test_loader):  # Loop through the eval set
                    if cuda:
                        geometry = geometry.cuda()
                        spectra = spectra.cuda()
                    logit = self.model_f(geometry)
                    loss = self.make_loss(logit, spectra)  # compute the loss
                    test_loss += loss  # Aggregate the loss

                # Record the testing loss to the tensorboard
                test_avg_loss = test_loss.cpu().data.numpy() / (j + 1)
                self.log.add_scalar('Loss/test', test_avg_loss, epoch)

                print("This is Epoch %d, training loss %.5f, validation loss %.5f" \
                      % (epoch, train_avg_loss, test_avg_loss))

                # Model improving, save the model down
                if test_avg_loss < self.best_validation_loss:
                    self.best_validation_loss = test_avg_loss
                    self.save_f()
                    print("Saving the model down...")

                    if self.best_validation_loss < self.flags.stop_threshold:
                        print("Training finished EARLIER at epoch %d, reaching loss of %.5f" % \
                              (epoch, self.best_validation_loss))
                        return None

    def train(self):
        """
        The major training function. This would start the training using information given in the flags
        :return: None
        """
        print("Start Discriminator and Generator Training now")
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.model_d.cuda()
            self.model_g.cuda()
            self.model_se.cuda()
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        # Construct optimizer after the model moved to GPU
        self.optm_d = self.make_optimizer_d()
        self.lr_scheduler = self.make_lr_scheduler(self.optm_d)
        self.optm_g = self.make_optimizer_g()
        self.lr_scheduler = self.make_lr_scheduler(self.optm_g)

        for epoch in range(self.flags.train_step):
            # Set to Training Mode
            train_loss_g = 0
            train_loss_d = 0
            # boundary_loss = 0                 # Unnecessary during training since we provide geometries
            self.model_d.train()
            self.model_g.train()
            self.model_se.train()
            for j, (geometry, spectra) in enumerate(self.train_loader):
                valid = torch.zeros(geometry.size(0), 1, requires_grad=False)
                if cuda:
                    geometry = geometry.cuda()  # Put data onto GPU
                    spectra = spectra.cuda()  # Put data onto GPU
                    valid = valid.cuda()
                """
                Adversarial Training starts, first train the generator part
                """
                self.optm_d.zero_grad()  # Zero the gradient first
                S_enc = self.model_se(spectra)  # Encode the spectra
                z = Tensor(np.random.normal(0, 1, (geometry.shape[0], self.flags.dim_z)))  # Create the noise
                geo_fake = self.model_g(S_enc, z)  # Generate the geometry_fake
                spec_fake = self.model_f(geo_fake)  # Get the resulting spectra
                fake_score = self.make_loss(spec_fake, spectra, G=geo_fake)  # Make loss
                fake_score.backward(retain_graph=True)  # Calculate the backward gradients
                self.optm_g.step()  # Move one step the optimizer
                train_loss_g += fake_score  # Aggregate the loss

                """
                Training the discriminator part
                """
                self.optm_d.zero_grad()
                # The real pairs
                loss_real = self.make_loss(self.model_d(geometry, S_enc), valid)
                # The fake pairs
                loss_fake = self.make_loss(self.model_d(geo_fake, S_enc), fake_score)
                d_loss = (loss_fake + loss_real) / 2
                d_loss.backward()
                self.optm_d.step()
                train_loss_d += d_loss  # Aggregate the loss

            # Calculate the avg loss of training
            train_avg_loss_g = train_loss_g.cpu().data.numpy() / (j + 1)
            train_avg_loss_d = train_loss_d.cpu().data.numpy() / (j + 1)

            if epoch % self.flags.eval_step == 0:  # For eval steps, do the evaluations and tensor board
                # Record the training loss to the tensorboard
                self.log.add_scalar('Loss/generator_train', train_avg_loss_g, epoch)
                self.log.add_scalar('Loss/discriminator_train', train_avg_loss_d, epoch)
                # self.log.add_scalar('Loss/BDY_train', boundary_avg_loss, epoch)

                # Set to Evaluation Mode
                self.model_d.eval()
                self.model_g.eval()
                self.model_se.eval()
                self.model_f.eval()

                print("Doing Evaluation on the model now")
                test_loss = 0
                for j, (geometry, spectra) in enumerate(self.test_loader):  # Loop through the eval set
                    if cuda:
                        geometry = geometry.cuda()
                        spectra = spectra.cuda()
                    S_enc = self.model_se(spectra)  # Encode the spectra
                    z = Tensor(np.random.normal(0, 1, (geometry.shape[0], self.flags.dim_z)))  # Create the noise
                    geo_fake = self.model_g(S_enc, z)  # Generate the geometry_fake
                    spec_fake = self.model_f(geo_fake)  # Get the resulting spectra
                    fake_score = self.make_loss(spec_fake, spectra, G=geo_fake)  # Make loss
                    test_loss += fake_score  # Aggregate the loss

                # Record the testing loss to the tensorboard
                test_avg_loss = test_loss.cpu().data.numpy() / (j + 1)
                self.log.add_scalar('Loss/generator_test', test_avg_loss, epoch)

                print("This is Epoch %d, generator training loss %.5f, discriminator training loss %.5f,"\
                      " validation loss %.5f" % (epoch, train_avg_loss_g, train_avg_loss_d, test_avg_loss))

                # Model improving, save the model down
                if test_avg_loss < self.best_validation_loss:
                    self.best_validation_loss = test_avg_loss
                    self.save_d()
                    self.save_g()
                    self.save_se()
                    print("Saving the model down...")

                    if self.best_validation_loss < self.flags.stop_threshold:
                        print("Training finished EARLIER at epoch %d, reaching loss of %.5f" % \
                              (epoch, self.best_validation_loss))
                        return None

            # Learning rate decay upon plateau
            self.lr_scheduler.step(train_avg_loss_g)
        self.log.close()

    def evaluate(self, save_dir='data/'):
        self.load()  # load the model as constructed
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.model.cuda()

        # Set to evaluation mode for batch_norm layers
        self.model.eval()
        self.model.bp = True

        # Construct optimizer after the model moved to GPU
        self.optm_eval = self.make_optimizer_eval()
        self.lr_scheduler = self.make_lr_scheduler(self.optm_eval)

        # Get the file names
        Ypred_file = os.path.join(save_dir, 'test_Ypred_{}.csv'.format(self.saved_model))
        Xtruth_file = os.path.join(save_dir, 'test_Xtruth_{}.csv'.format(self.saved_model))
        Ytruth_file = os.path.join(save_dir, 'test_Ytruth_{}.csv'.format(self.saved_model))
        Xpred_file = os.path.join(save_dir, 'test_Xpred_{}.csv'.format(self.saved_model))

        # Open those files to append
        with open(Xtruth_file, 'a') as fxt, open(Ytruth_file, 'a') as fyt, \
                open(Ypred_file, 'a') as fyp, open(Xpred_file, 'a') as fxp:
            # Loop through the eval data and evaluate
            for ind, (geometry, spectra) in enumerate(self.test_loader):
                if cuda:
                    geometry = geometry.cuda()
                    spectra = spectra.cuda()
                # Initialize the geometry first
                self.model.randomize_geometry_eval()
                self.optm_eval = self.make_optimizer_eval()
                self.lr_scheduler = self.make_lr_scheduler(self.optm_eval)
                Xpred, Ypred = self.evaluate_one(spectra)
                np.savetxt(fxt, geometry.cpu().data.numpy(), fmt='%.3f')
                np.savetxt(fyt, spectra.cpu().data.numpy(), fmt='%.3f')
                np.savetxt(fyp, Ypred, fmt='%.3f')
                np.savetxt(fxp, Xpred, fmt='%.3f')
        return Ypred_file, Ytruth_file

    def evaluate_one(self, target_spectra):
        # expand the target spectra to eval batch size
        target_spectra_expand = target_spectra.expand([self.flags.eval_batch_size, -1])
        # Start backprop
        for i in range(self.flags.eval_step):
            logit = self.model(self.model.geometry_eval)  # Get the output
            loss = self.make_loss(logit, target_spectra_expand)  # Get the loss
            loss.backward()  # Calculate the Gradient
            self.optm_eval.step()  # Move one step the optimizer

            # check periodically to stop and print stuff
            if i % self.flags.verb_step == 0:
                print("loss at inference step{} : {}".format(i, loss.data))  # Print loss
                if loss.data < self.flags.stop_threshold:  # Check if stop
                    print("Loss is lower than threshold{}, inference stop".format(self.flags.stop_threshold))
                    break

        # Get the best performing one
        best_estimate_index = np.argmin(loss.cpu().data.numpy())
        Xpred_best = self.model.geometry_eval.cpu().data.numpy()[best_estimate_index, :]
        Ypred_best = logit.cpu().data.numpy()[best_estimate_index, :]

        return Xpred_best, Ypred_best
