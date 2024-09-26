import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from dataset import load_img
import os
from models.model import *

from misc.metric_tool import ConfuseMatrixMeter

from misc.logger_tool import Logger, Timer
import torch.nn.functional as F


class CDEvaluator():

    def __init__(self, dataloader):

        self.dataloader = dataloader

        self.n_class = 2
        # define G
        self.net_G = Net()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #self.net_G = self.net_G.to(self.device)

        #print(self.device)

        # define some other vars to record the training states
        self.running_metric = ConfuseMatrixMeter(n_class=self.n_class)

        self.checkpoint_dir = "checkpoints/"

        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)

        # define logger file
        logger_path = os.path.join(self.checkpoint_dir, 'log_test.txt')
        self.logger = Logger(logger_path)

        #  training log
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0

        self.steps_per_epoch = len(dataloader)

        self.G_pred = None
        self.pred_vis = None
        self.batch = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0

    def _load_checkpoint(self, checkpoint_name='best.pt'):

        if os.path.exists(os.path.join(self.checkpoint_dir, checkpoint_name)):
            self.logger.write('loading last checkpoint...\n')
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, checkpoint_name), map_location=self.device)

            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])

            self.net_G.to(self.device)

            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.logger.write('Eval Historical_best_acc = %.4f (at epoch %d)\n' %
                              (self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')

        else:
            raise FileNotFoundError('no such checkpoint %s' % checkpoint_name)

    def _update_metric(self, labels):
        """
        update metric
        """
        target = labels.to(self.device).detach()

        G_pred = self.G_pred.detach()

        G_pred = torch.argmax(G_pred, dim=1)

        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())

        return current_score

    def _collect_running_batch_states(self, labels):

        running_acc = self._update_metric(labels)

        m = len(self.dataloader)

        if np.mod(self.batch_id, 100) == 1:
            message = 'Is_training: %s. [%d,%d],  running_mf1: %.5f\n' %\
                      (self.is_training, self.batch_id, m, running_acc)
            self.logger.write(message)


    def _collect_epoch_states(self):

        scores_dict = self.running_metric.get_scores()

        # np.save(os.path.join(self.checkpoint_dir, 'scores_dict.npy'), scores_dict)

        # self.epoch_acc = scores_dict['mf1']
        #
        # with open(os.path.join(self.checkpoint_dir, '%s.txt' % (self.epoch_acc)),
        #           mode='a') as file:
        #     pass

        message = ''
        for k, v in scores_dict.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write('%s\n' % message)  # save the message

        self.logger.write('\n')

    def _clear_cache(self):
        self.running_metric.clear()

    def _forward_pass(self, imageA, imageB):

        img_in1 = imageA.to(self.device)
        img_in2 = imageB.to(self.device)
        self.G_pred = self.net_G(img_in1, img_in2)

    def _handle_label(self, labels):
        labels = labels.to(self.device)
        labels[labels >= 0.5] = 1
        labels[labels < 0.5] = 0
        return labels

    def eval_models(self, checkpoint_name='best.pt'):

        self._load_checkpoint(checkpoint_name)

        ################## Eval ##################
        ##########################################
        self.logger.write('Begin evaluation...\n')
        self._clear_cache()
        self.is_training = False
        self.net_G.eval()

        for self.batch_id, (imageA, imageB, labels) in enumerate(self.dataloader, 0):
            labels = self._handle_label(labels)
            with torch.no_grad():
                self._forward_pass(imageA, imageB)
            self._collect_running_batch_states(labels)
        self._collect_epoch_states()

if __name__ == '__main__':

    train_dataloader, test_dataloader, val_dataloader, batch_size = load_img()
    model = CDEvaluator(test_dataloader)
    model.eval_models()

