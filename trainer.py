import numpy as np
import torch.optim as optim
from dataset import load_img
import os

from models.model import *

from misc.metric_tool import ConfuseMatrixMeter
from models.myloss import CombinedLoss, MyEntropyLoss
import time

from misc.logger_tool import Logger, Timer


class CDTrainer():

    def __init__(self, train_dataloader, val_dataloader, batch_size):

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.n_class = 2
        # define G
        self.net_G = Net()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.net_G = self.net_G.to(self.device)

        print(self.device)

        self.optimizer_G = optim.SGD(self.net_G.parameters(), lr=0.002,
                                            momentum=0.98,
                                            weight_decay=5e-4)

        self.exp_lr_scheduler_G = optim.lr_scheduler.StepLR(self.optimizer_G, step_size=10, gamma=0.7)

        self.running_metric = ConfuseMatrixMeter(n_class=2)

        self.checkpoint_dir = "checkpoints/"

        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)

        logger_path = os.path.join(self.checkpoint_dir, 'log.txt')
        self.logger = Logger(logger_path)

        self.timer = Timer()
        self.batch_size = batch_size
        self.begin_total = time.time()  # 开始时间
        self.end_total = 0

        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0
        self.epoch_to_start = 0
        self.max_num_epochs = 150

        self.global_step = 0
        self.steps_per_epoch = len(self.train_dataloader)
        self.total_steps = (self.max_num_epochs - self.epoch_to_start)*self.steps_per_epoch

        self.G_pred = None
        self.pred_vis = None
        self.batch = None
        self.G_loss = 0
        self.loss1 = 0
        self.loss2 = 0
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self._pxl_loss = CombinedLoss()
        self.my_loss = MyEntropyLoss()
    def _load_checkpoint(self, ckpt_name='last.pt'):

        if os.path.exists(os.path.join(self.checkpoint_dir, ckpt_name)):
            self.logger.write('loading last checkpoint...\n')
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, ckpt_name),
                                    map_location=self.device)
            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])

            self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            self.exp_lr_scheduler_G.load_state_dict(
                checkpoint['exp_lr_scheduler_G_state_dict'])

            self.net_G.to(self.device)

            self.epoch_to_start = checkpoint['epoch_id'] + 1
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.total_steps = (self.max_num_epochs - self.epoch_to_start)*self.steps_per_epoch

            self.logger.write('Epoch_to_start = %d, Historical_best_acc = %.4f (at epoch %d)\n' %
                  (self.epoch_to_start, self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')

        else:
            print('training from scratch...')

    def _timer_update(self):
        self.global_step = (self.epoch_id-self.epoch_to_start) * self.steps_per_epoch + self.batch_id

        self.timer.update_progress((self.global_step + 1) / self.total_steps)
        est = self.timer.estimated_remaining()
        imps = (self.global_step + 1) * self.batch_size / self.timer.get_stage_elapsed()
        return imps, est

    def _save_checkpoint(self, ckpt_name):
        torch.save({
            'epoch_id': self.epoch_id,
            'best_val_acc': self.best_val_acc,
            'best_epoch_id': self.best_epoch_id,
            'model_G_state_dict': self.net_G.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'exp_lr_scheduler_G_state_dict': self.exp_lr_scheduler_G.state_dict(),
        }, os.path.join(self.checkpoint_dir, ckpt_name))

    def _update_lr_schedulers(self):
        self.exp_lr_scheduler_G.step()

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

        m = len(self.train_dataloader)
        if self.is_training is False:
            m = len(self.val_dataloader)

        imps, est = self._timer_update()

        if np.mod(self.batch_id, 100) == 1:
            message = 'Is_training: %s. [%d,%d][%d,%d], imps: %.2f, est: %.2fh, G_loss: %.5f, running_mf1: %.5f\n' %\
                      (self.is_training, self.epoch_id, self.max_num_epochs-1, self.batch_id, m,
                     imps*self.batch_size, est,
                     self.G_loss.item(), running_acc)
            self.logger.write(message)



    def _collect_epoch_states(self):
        scores = self.running_metric.get_scores()
        self.epoch_acc = scores['mf1']
        self.logger.write('Is_training: %s. Epoch %d / %d, epoch_mF1= %.5f\n' %
              (self.is_training, self.epoch_id, self.max_num_epochs-1, self.epoch_acc))
        message = ''
        for k, v in scores.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write(message+'\n')
        self.logger.write('\n')

    def _update_checkpoints(self):

        self._save_checkpoint(ckpt_name='last.pt')
        self.logger.write('Lastest model updated. Epoch_acc=%.4f, Historical_best_acc=%.4f (at epoch %d)\n'
              % (self.epoch_acc, self.best_val_acc, self.best_epoch_id))
        self.logger.write('\n')

        if self.epoch_acc > self.best_val_acc:
            self.best_val_acc = self.epoch_acc
            self.best_epoch_id = self.epoch_id
            self._save_checkpoint(ckpt_name='best.pt')
            self.logger.write('*' * 10 + 'Best model updated!\n')
            self.logger.write('\n')

    def _clear_cache(self):
        self.running_metric.clear()


    def _forward_pass(self, imageA, imageB):

        img_in1 = imageA.to(self.device)
        img_in2 = imageB.to(self.device)
        self.G_pred = self.net_G(img_in1, img_in2)


    def _backward_G(self, labels):

        gt = labels.to(self.device)
        self.loss1 = self._pxl_loss(self.G_pred, gt)
        self.loss2 = self.my_loss(self.G_pred, gt)
        self.G_loss = 15 * self.loss1 + 1 * self.loss2
        self.G_loss.backward()

    def train_models(self):

        self._load_checkpoint()

        # loop over the dataset multiple times
        for self.epoch_id in range(self.epoch_to_start, self.max_num_epochs):

            ################## train #################
            ##########################################

            self.logger.write('\nEpoch: %d' % (self.epoch_id + 1))
            end = time.time()  # 每个epoch开始时间
            hua = end - self.begin_total
            self.logger.write(f'\nafter：{hua // 3600} Hour {(hua % 3600) // 60} Minute {int(hua % 60)} second ')  # 每个epoch开始的绝对时间

            self._clear_cache()
            self.is_training = True
            self.net_G.train()
            self.logger.write('\nlr: %0.7f\n' % self.optimizer_G.param_groups[0]['lr'])
            for self.batch_id, (imageA, imageB, labels) in enumerate(self.train_dataloader, 0):
                self._forward_pass(imageA, imageB)
                # update G
                self.optimizer_G.zero_grad()
                self._backward_G(labels)
                self.optimizer_G.step()
                self._collect_running_batch_states(labels)
                self._timer_update()

            self._collect_epoch_states()
            self._update_lr_schedulers()

            ################## Eval ##################
            ##########################################
            self.logger.write('Begin evaluation...\n')
            self._clear_cache()
            self.is_training = False
            self.net_G.eval()

            # Iterate over data.
            for self.batch_id, (imageA, imageB, labels) in enumerate(self.val_dataloader, 0):
                with torch.no_grad():
                    self._forward_pass(imageA, imageB)
                self._collect_running_batch_states(labels)
            self._collect_epoch_states()

            ########### Update_Checkpoints ###########
            ##########################################
            self._update_checkpoints()

            end_these = time.time()  # 每个epoch结束时间
            hua1 = end_these - end
            self.logger.write(
                f'\nthis epoch time：{hua1 // 3600} hour {(hua1 % 3600) // 60} minute {int(hua1 % 60)} second\n')  # 每个epoch需要的时间

        self.end_total = time.time()  # 总训练完的绝对时间
        hua2 = self.end_total - self.begin_total
        self.logger.write(f'\ntotal time：{hua2 // 3600} hour {(hua2 % 3600) // 60} minute {int(hua2 % 60)} second\n')  # 总训练完的相对时间

if __name__ == '__main__':
    train_dataloader, test_dataloader, val_dataloader, batch_size = load_img()

    model = CDTrainer(train_dataloader, val_dataloader, batch_size)
    model.train_models()