import os
import os.path as osp

import torch
import torch.optim as optim
import tqdm
from glog import logger

from adapters import get_model_adapter
from losses.loss import get_loss
from metric_utils import EarlyStopping, get_metric_counter


class Trainer(object):
    """ Model trainer
    """

    def __init__(self, model, config, train, val):
        """
        Args:
            model: model to be trained
            config (dict): dictionary with:
                - warmup_num (int): number of warmup epochs
                - steps_per_epoch (int): training steps per epoch
                - validation_steps (int): validation steps per epoch
                - model (dict): model config
                - optimizer (dict): optimizer config
                - scheduler (dict): scheduler config
            train (torch.utils.data.DataLoader): training dataloader
            val (torch.utils.data.DataLoader): validation dataloader
        """
        self.config = config
        self.model = model
        self.train_dataset = train
        self.val_dataset = val
        self.warmup_epochs = config.get('warmup_num', 0)
        self.metric_counter = get_metric_counter(config)
        self.steps_per_epoch = config.get("steps_per_epoch", len(self.train_dataset))
        self.validation_steps = config.get('validation_steps', len(self.val_dataset))
        self.scheduler_backbone = None  # Initialize the scheduler for the backbone
        self.scheduler_appended = None  # Initialize the scheduler for the appended layers

    def train(self):
        """ Trains model for a given number of epochs.
        Model is trained, evaluated and saved on every epoch
        """
        self._init_params()
        for epoch in range(0, self.config['num_epochs']):
            if (epoch == self.warmup_epochs) and not (self.warmup_epochs == 0):
                self.model.module.unfreeze()
                self.optimizer = self._get_optim(self.model.parameters())
                self.scheduler = self._get_scheduler(self.optimizer)
            self._run_epoch(epoch)
            self._validate(epoch)
            self._update_scheduler()
            self._set_checkpoint()
            self.early_stopping(val_metric=self.metric_counter.get_metric())
            if self.early_stopping.early_stop:
                break

    def _set_checkpoint(self):
        """ Saves model weights in the last checkpoint.
        Also, model is saved as the best model if model has the best metric
        """
        if self.metric_counter.update_best_model():
            torch.save({
                'model': self.model_adapter.get_model_export(self.model)
            }, osp.join(self.config['experiment']['folder'], self.config['experiment']['name'], 'best.h5'))
        torch.save({
            'model': self.model_adapter.get_model_export(self.model)
        }, osp.join(self.config['experiment']['folder'], self.config['experiment']['name'], 'last.h5'))
        logger.info(self.metric_counter.loss_message())

    def _run_epoch(self, epoch):
        """ Runs one training epoch
        Args:
            epoch (int): epoch number
        """
        self.model = self.model.train()
        self.metric_counter.clear()
        # lr = self.optimizer.param_groups[0]['lr']
        # tq = tqdm.tqdm(total=self.steps_per_epoch)
        # tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        lr_backbone = self.optimizer_backbone.param_groups[0]['lr']  # Learning rate for the backbone
        if hasattr(self, 'optimizer_appended') and self.optimizer_appended is not None:
            lr_appended = self.optimizer_appended.param_groups[0]['lr']
        tq = tqdm.tqdm(total=self.steps_per_epoch)
        if hasattr(self, 'optimizer_appended') and self.optimizer_appended is not None:
            tq.set_description('Epoch {}, lr_backbone {}, lr_appended {}'.format(epoch, lr_backbone, lr_appended))
        else:
            tq.set_description('Epoch {}, lr_backbone {}'.format(epoch, lr_backbone))
        for i, data in enumerate(self.train_dataset):
            images, targets = self.model_adapter.get_input(data)
            outputs = self.model(images)
            self.optimizer_backbone.zero_grad()
            if hasattr(self, 'optimizer_appended') and self.optimizer_appended is not None:
                self.optimizer_appended.zero_grad()
            
            loss = self.criterion(outputs, targets)
            total_loss, loss_dict = self.model_adapter.get_loss(loss)
            total_loss.backward()
            
            self.optimizer_backbone.step()
            if hasattr(self, 'optimizer_appended') and self.optimizer_appended is not None:
                self.optimizer_appended.step()
            
            self.metric_counter.add_losses(loss_dict)
            tq.update()
            tq.set_postfix(loss=self.metric_counter.loss_message())
            
            if i >= self.steps_per_epoch:
                break
        tq.close()
        self.metric_counter.write_to_tensorboard(epoch)

    def _validate(self, epoch):
        """ Performs validation of current model
        Args:
            epoch (int): epoch number
        """
        self.model = self.model.eval()
        with torch.no_grad():
            self.metric_counter.clear()
            for i, data in enumerate(self.val_dataset):
                images, targets = self.model_adapter.get_input(data)
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                _, loss_dict = self.model_adapter.get_loss(loss)
                self.metric_counter.add_losses(loss_dict)

                # calculate metrics
                metrics = self.model_adapter.get_metrics(outputs, targets)
                self.metric_counter.add_metrics(metrics)

                if i >= self.validation_steps:
                    break
            self.metric_counter.write_to_tensorboard(epoch, validation=True)

    def _get_optim(self, params):
        """ Creates model optimizer from Trainer config
        Args:
            params (list): list of model parameters to be trained
        Returns:
            torch.optim.optimizer.Optimizer: model optimizer
        """
        optimizer_config = self.config['optimizer']
        if optimizer_config['name'] == 'adam':
            optimizer = optim.Adam(params, lr=optimizer_config['lr'])
        elif optimizer_config['name'] == 'sgd':
            optimizer = optim.SGD(params,
                                  lr=optimizer_config['lr'],
                                  momentum=optimizer_config.get('momentum', 0),
                                  weight_decay=optimizer_config.get('weight_decay', 0))
        elif optimizer_config['name'] == 'adadelta':
            optimizer = optim.Adadelta(params, lr=optimizer_config['lr'])
        else:
            raise ValueError("Optimizer [%s] not recognized." % optimizer_config['name'])
        return optimizer

    def _update_scheduler(self):
        """ Updates scheduler
        """
        # if self.config['scheduler']['name'] == 'plateau':
        #     self.scheduler.step(self.metric_counter.get_metric())
        # else:
        #     self.scheduler.step()
        # Step the learning rate scheduler for the backbone optimizer
        self.scheduler_backbone.step(self.metric_counter.get_metric())
        if hasattr(self, 'scheduler_appended') and self.scheduler_appended is not None:
            # Step the learning rate scheduler for the appended optimizer
            self.scheduler_appended.step(self.metric_counter.get_metric())

    def _get_scheduler(self, optimizer):
        """ Creates scheduler for a given optimizer from Trainer config
        Args:
            optimizer (torch.optim.optimizer.Optimizer): optimizer to be updated
        Returns:
            torch.optim.lr_scheduler._LRScheduler: optimizer scheduler
        """
        scheduler_config = self.config['scheduler']
        if scheduler_config['name'] == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                             mode=scheduler_config['mode'],
                                                             patience=scheduler_config['patience'],
                                                             factor=scheduler_config['factor'],
                                                             min_lr=scheduler_config['min_lr'])
        elif scheduler_config['name'] == 'MultiStepLR':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                       scheduler_config['milestones'],
                                                       gamma=scheduler_config['gamma'])
        else:
            raise ValueError("Scheduler [%s] not recognized." % scheduler_config['name'])
        return scheduler

    def _init_params(self):
        """ Initializes Trainer
        Initialized attributes:
            - criterion: loss to be used during training
            - optimizer: model optimizer
            - scheduler: scheduler for optimizer
            - early_stopping: Early Stopping technique, which stops training if validation loss doesn't improve
               after a given patience
            - model_adapter: adapter for a given model
        """
        self.criterion = get_loss(self.config['model']['loss'])
        optimizer_config = self.config['optimizer']
        # self.optimizer = self._get_optim(filter(lambda p: p.requires_grad, self.model.parameters()))
        # self.scheduler = self._get_scheduler(self.optimizer)
        # Separate the parameters into two groups: backbone and appended layers
        backbone_params = []
        appended_params = []
        
        # for mobilenet
        appended_layers = ["module.fpn.backbones.0.1","module.fpn.backbones.1.1","module.fpn.backbones.2.1",
                           "module.fpn.backbones.3.1","module.fpn.backbones.4.1","module.fpn.backbones.4.2"]
        
        # for shufflenet
        appended_layers = ["module.fpn.backbones.4.1"]
        for name, param in self.model.named_parameters():
            appended= False
            print(name)
            for element in appended_layers:
                if element in name:
                    appended_params.append(param)
                    print("appended layer: ", name)
                    appended=True
                    break
            if not appended:
                backbone_params.append(param)

        # Define different learning rates for each group
        backbone_lr = optimizer_config['lr'] # Adjust this value as needed
        appended_lr = optimizer_config['appended_lr']   # Adjust this value as needed

        self.optimizer_backbone = optim.Adam(backbone_params, lr=backbone_lr)
        self.scheduler_backbone = self._get_scheduler(self.optimizer_backbone)
        
        # self.scheduler = self._get_scheduler(self.optimizer_backbone)  # You can choose either optimizer for scheduling
        
        if len(appended_params) != 0:
            self.optimizer_appended = optim.Adam(appended_params, lr=appended_lr)
            self.scheduler_appended = self._get_scheduler(self.optimizer_appended)

        self.early_stopping = EarlyStopping(patience=self.config['early_stopping'])
        self.model_adapter = get_model_adapter(self.config)
        os.makedirs(osp.join(self.config['experiment']['folder'], self.config['experiment']['name']), exist_ok=True)
