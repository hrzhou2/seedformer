# --------------------------------------------------------
# SeedFormer
# Copyright (c) 2022 Intelligent Media Pervasive, Recognition & Understanding Lab of NJU - All Rights Reserved
# Licensed under The MIT License [see LICENSE for details]
# Written by Haoran Zhou
# --------------------------------------------------------

'''
==============================================================

SeedFormer: Point Cloud Completion
-> Training/Testing Manager

==============================================================

Author: Haoran Zhou
Date: 2022-5-31

==============================================================
'''


import os
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
import time

import utils.data_loaders
import utils.helpers
from utils.average_meter import AverageMeter
from utils.metrics import Metrics
from utils.schedular import GradualWarmupScheduler
from utils.loss_utils import get_loss
from utils.ply import read_ply, write_ply
import pointnet_utils.pc_util as pc_util
from PIL import Image


# ----------------------------------------------------------------------------------------------------------------------
#
#           Trainer
#       \***************/
#

class Manager:

    # Initialization methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, model, cfg):
        """
        Initialize parameters and start training/testing
        :param model: network object
        :param cfg: configuration object
        """

        ############
        # Parameters
        ############
        
        # training dataset
        self.dataset = cfg.DATASET.TRAIN_DATASET

        # Epoch index
        self.epoch = 0

        # Create the optimizers
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                           lr=cfg.TRAIN.LEARNING_RATE,
                                           weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                                           betas=cfg.TRAIN.BETAS)

        # lr scheduler
        self.scheduler_steplr = StepLR(self.optimizer, step_size=1, gamma=0.1 ** (1 / cfg.TRAIN.LR_DECAY))
        self.lr_scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=cfg.TRAIN.WARMUP_EPOCHS,
                                              after_scheduler=self.scheduler_steplr)

        # record file
        self.train_record_file = open(os.path.join(cfg.DIR.LOGS, 'training.txt'), 'w')
        self.test_record_file = open(os.path.join(cfg.DIR.LOGS, 'testing.txt'), 'w')

        # eval metric
        self.best_metrics = float('inf')
        self.best_epoch = 0


    # Record functions
    def train_record(self, info, show_info=True):
        if show_info:
            print(info)
        if self.train_record_file:
            self.train_record_file.write(info + '\n')
            self.train_record_file.flush()

    def test_record(self, info, show_info=True):
        if show_info:
            print(info)
        if self.test_record_file:
            self.test_record_file.write(info + '\n')
            self.test_record_file.flush()

    def unpack_data(self, data):

        if self.dataset == 'ShapeNet':
            partial = data['partial_cloud']
            gt = data['gtcloud']
        elif self.dataset == 'ShapeNet55':
            # generate partial data online
            gt = data['gtcloud']
            _, npoints, _ = gt.shape
            partial, _ = utils.helpers.seprate_point_cloud(gt, npoints, [int(npoints * 1/4) , int(npoints * 3/4)], fixed_points = None)
        else:
            raise ValueError('No method implemented for this dataset: {:s}'.format(self.dataset))

        return partial, gt


    def train(self, model, train_data_loader, val_data_loader, cfg):

        init_epoch = 0
        steps = 0

        # training record file
        print('Training Record:')
        self.train_record('n_itr, cd_pc, cd_p1, cd_p2, cd_p3, partial_matching')
        print('Testing Record:')
        self.test_record('#epoch cdc cd1 cd2 partial_matching | cd3 | #best_epoch best_metrics')

        # Training Start
        for epoch_idx in range(init_epoch + 1, cfg.TRAIN.N_EPOCHS + 1):

            self.epoch = epoch_idx

            # timer
            epoch_start_time = time.time()

            model.train()

            # Update learning rate
            self.lr_scheduler.step()

            # total cds
            total_cd_pc = 0
            total_cd_p1 = 0
            total_cd_p2 = 0
            total_cd_p3 = 0
            total_partial = 0

            batch_end_time = time.time()
            n_batches = len(train_data_loader)
            learning_rate = self.optimizer.param_groups[0]['lr']
            for batch_idx, (taxonomy_ids, model_ids, data) in enumerate(train_data_loader):
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)

                # unpack data
                partial, gt = self.unpack_data(data)

                pcds_pred = model(partial)

                loss_total, losses, gts = get_loss(pcds_pred, partial, gt, sqrt=True)

                self.optimizer.zero_grad()
                loss_total.backward()
                self.optimizer.step()

                cd_pc_item = losses[0].item() * 1e3
                total_cd_pc += cd_pc_item
                cd_p1_item = losses[1].item() * 1e3
                total_cd_p1 += cd_p1_item
                cd_p2_item = losses[2].item() * 1e3
                total_cd_p2 += cd_p2_item
                cd_p3_item = losses[3].item() * 1e3
                total_cd_p3 += cd_p3_item
                partial_item = losses[4].item() * 1e3
                total_partial += partial_item
                n_itr = (epoch_idx - 1) * n_batches + batch_idx

                # training record
                message = '{:d} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(n_itr, cd_pc_item, cd_p1_item, cd_p2_item, cd_p3_item, partial_item)
                self.train_record(message, show_info=False)

            # avg cds
            avg_cdc = total_cd_pc / n_batches
            avg_cd1 = total_cd_p1 / n_batches
            avg_cd2 = total_cd_p2 / n_batches
            avg_cd3 = total_cd_p3 / n_batches
            avg_partial = total_partial / n_batches

            epoch_end_time = time.time()

            # Training record
            self.train_record(
                '[Epoch %d/%d] LearningRate = %f EpochTime = %.3f (s) Losses = %s' %
                (epoch_idx, cfg.TRAIN.N_EPOCHS, learning_rate, epoch_end_time - epoch_start_time, ['%.4f' % l for l in [avg_cdc, avg_cd1, avg_cd2, avg_cd3, avg_partial]]))

            # Validate the current model
            cd_eval = self.validate(cfg, model=model, val_data_loader=val_data_loader)
            self.train_record('Testing scores = {:.4f}'.format(cd_eval))

            # Save checkpoints
            if cd_eval < self.best_metrics:
                self.best_epoch = epoch_idx
                file_name = 'ckpt-best.pth' if cd_eval < self.best_metrics else 'ckpt-epoch-%03d.pth' % epoch_idx
                output_path = os.path.join(cfg.DIR.CHECKPOINTS, file_name)
                torch.save({
                    'epoch_index': epoch_idx,
                    'best_metrics': cd_eval,
                    'model': model.state_dict()
                }, output_path)

                print('Saved checkpoint to %s ...' % output_path)
                if cd_eval < self.best_metrics:
                    self.best_metrics = cd_eval

        # training end
        self.train_record_file.close()
        self.test_record_file.close()


    def validate(self, cfg, model=None, val_data_loader=None, outdir=None):
        # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
        torch.backends.cudnn.benchmark = True

        # Switch models to evaluation mode
        model.eval()

        n_samples = len(val_data_loader)
        test_losses = AverageMeter(['cdc', 'cd1', 'cd2', 'cd3', 'partial_matching'])
        test_metrics = AverageMeter('cd3')

        # Start testing
        for model_idx, (taxonomy_id, model_id, data) in enumerate(val_data_loader):
            taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
            #model_id = model_id[0]

            with torch.no_grad():
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)

                # unpack data
                partial, gt = self.unpack_data(data)

                # forward
                pcds_pred = model(partial.contiguous())
                loss_total, losses, _ = get_loss(pcds_pred, partial, gt, sqrt=True)

                # get metrics
                cdc = losses[0].item() * 1e3
                cd1 = losses[1].item() * 1e3
                cd2 = losses[2].item() * 1e3
                cd3 = losses[3].item() * 1e3
                partial_matching = losses[4].item() * 1e3

                _metrics = [cd3]
                test_losses.update([cdc, cd1, cd2, cd3, partial_matching])
                test_metrics.update(_metrics)

        # Record testing results
        message = '#{:d} {:.4f} {:.4f} {:.4f} {:.4f} | {:.4f} | #{:d} {:.4f}'.format(self.epoch, test_losses.avg(0), test_losses.avg(1), test_losses.avg(2), test_losses.avg(4), test_losses.avg(3), self.best_epoch, self.best_metrics)
        self.test_record(message. show_info=False)

        return test_losses.avg(3)

    def test(self, cfg, model, test_data_loader, outdir, mode=None):

        if self.dataset == 'ShapeNet':
            self.test_pcn(cfg, model, test_data_loader, outdir)
        elif self.dataset == 'ShapeNet55':
            self.test_shapenet55(cfg, model, test_data_loader, outdir, mode)
        else:
            raise ValueError('No testing method implemented for this dataset: {:s}'.format(self.dataset))

    def test_pcn(self, cfg, model=None, test_data_loader=None, outdir=None):
        """
        Testing Method for dataset PCN
        """

        # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
        torch.backends.cudnn.benchmark = True

        # Switch models to evaluation mode
        model.eval()

        n_samples = len(test_data_loader)
        test_losses = AverageMeter(['cdc', 'cd1', 'cd2', 'cd3', 'partial_matching'])
        test_metrics = AverageMeter(Metrics.names())
        mclass_metrics = AverageMeter(Metrics.names())
        category_metrics = dict()

        # Start testing
        for model_idx, (taxonomy_id, model_id, data) in enumerate(test_data_loader):
            taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
            #model_id = model_id[0]

            with torch.no_grad():
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)

                # unpack data
                partial, gt = self.unpack_data(data)

                # forward
                pcds_pred = model(partial.contiguous())
                loss_total, losses, _ = get_loss(pcds_pred, partial, gt, sqrt=True)

                # get loss
                cdc = losses[0].item() * 1e3
                cd1 = losses[1].item() * 1e3
                cd2 = losses[2].item() * 1e3
                cd3 = losses[3].item() * 1e3
                partial_matching = losses[4].item() * 1e3
                test_losses.update([cdc, cd1, cd2, cd3, partial_matching])

                # get all metrics
                _metrics = Metrics.get(pcds_pred[-1], gt)
                test_metrics.update(_metrics)
                if taxonomy_id not in category_metrics:
                    category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[taxonomy_id].update(_metrics)

                # output to file
                if outdir:
                    if not os.path.exists(os.path.join(outdir, taxonomy_id)):
                        os.makedirs(os.path.join(outdir, taxonomy_id))
                    if not os.path.exists(os.path.join(outdir, taxonomy_id+'_images')):
                        os.makedirs(os.path.join(outdir, taxonomy_id+'_images'))
                    # save pred, gt, partial pcds 
                    pred = pcds_pred[-1]
                    for mm, model_name in enumerate(model_id):
                        output_file = os.path.join(outdir, taxonomy_id, model_name)
                        write_ply(output_file + '_pred.ply', pred[mm, :].detach().cpu().numpy(), ['x', 'y', 'z'])
                        write_ply(output_file + '_gt.ply', gt[mm, :].detach().cpu().numpy(), ['x', 'y', 'z'])
                        write_ply(output_file + '_partial.ply', partial[mm, :].detach().cpu().numpy(), ['x', 'y', 'z'])
                        # output img files
                        img_filename = os.path.join(outdir, taxonomy_id+'_images', model_name+'.jpg')
                        output_img = pc_util.point_cloud_three_views(pred[mm, :].detach().cpu().numpy(), diameter=7)
                        output_img = (output_img*255).astype('uint8')
                        im = Image.fromarray(output_img)
                        im.save(img_filename)


        # Record category results
        self.train_record('============================ TEST RESULTS ============================')
        self.train_record('Taxonomy\t#Sample\t' + '\t'.join(test_metrics.items))

        for taxonomy_id in category_metrics:
            message = '{:s}\t{:d}\t'.format(taxonomy_id, category_metrics[taxonomy_id].count(0)) 
            message += '\t'.join(['%.4f' % value for value in category_metrics[taxonomy_id].avg()])
            mclass_metrics.update(category_metrics[taxonomy_id].avg())
            self.train_record(message)

        self.train_record('Overall\t{:d}\t'.format(test_metrics.count(0)) + '\t'.join(['%.4f' % value for value in test_metrics.avg()]))
        self.train_record('MeanClass\t\t' + '\t'.join(['%.4f' % value for value in mclass_metrics.avg()]))

        # record testing results
        message = '#{:d} {:.4f} {:.4f} {:.4f} {:.4f} | {:.4f} | #{:d} {:.4f}'.format(self.epoch, test_losses.avg(0), test_losses.avg(1), test_losses.avg(2), test_losses.avg(4), test_losses.avg(3), self.best_epoch, self.best_metrics)
        self.test_record(message)


        return test_losses.avg(3)

    def test_shapenet55(self, cfg, model=None, test_data_loader=None, outdir=None, mode=None):
        """
        Testing Method for dataset shapenet-55/34
        """

        from models.utils import fps_subsample
        
        # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
        torch.backends.cudnn.benchmark = True

        # Eval settings
        crop_ratio = {
            'easy': 1/4,
            'median' :1/2,
            'hard':3/4
        }
        choice = [torch.Tensor([1,1,1]),torch.Tensor([1,1,-1]),torch.Tensor([1,-1,1]),torch.Tensor([-1,1,1]),
                  torch.Tensor([-1,-1,1]),torch.Tensor([-1,1,-1]), torch.Tensor([1,-1,-1]),torch.Tensor([-1,-1,-1])]

        # Switch models to evaluation mode
        model.eval()

        n_samples = len(test_data_loader)
        test_losses = AverageMeter(['cdc', 'cd1', 'cd2', 'cd3', 'partial_matching'])
        test_metrics = AverageMeter(Metrics.names())
        mclass_metrics = AverageMeter(Metrics.names())
        category_metrics = dict()

        # Start testing
        print('Start evaluating (mode: {:s}) ...'.format(mode))
        for model_idx, (taxonomy_id, model_id, data) in enumerate(test_data_loader):
            taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
            #model_id = model_id[0]

            with torch.no_grad():
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)

                # generate partial data online
                gt = data['gtcloud']
                _, npoints, _ = gt.shape
                
                # partial clouds from fixed viewpoints
                num_crop = int(npoints * crop_ratio[mode])
                for partial_id, item in enumerate(choice):
                    partial, _ = utils.helpers.seprate_point_cloud(gt, npoints, num_crop, fixed_points = item)
                    partial = fps_subsample(partial, 2048)

                    pcds_pred = model(partial.contiguous())
                    loss_total, losses, _ = get_loss(pcds_pred, partial, gt, sqrt=False) # L2

                    # get loss
                    cdc = losses[0].item() * 1e3
                    cd1 = losses[1].item() * 1e3
                    cd2 = losses[2].item() * 1e3
                    cd3 = losses[3].item() * 1e3
                    partial_matching = losses[4].item() * 1e3
                    test_losses.update([cdc, cd1, cd2, cd3, partial_matching])

                    # get all metrics
                    _metrics = Metrics.get(pcds_pred[-1], gt)
                    test_metrics.update(_metrics)
                    if taxonomy_id not in category_metrics:
                        category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                    category_metrics[taxonomy_id].update(_metrics)

                    # output to file
                    if outdir:
                        if not os.path.exists(os.path.join(outdir, taxonomy_id)):
                            os.makedirs(os.path.join(outdir, taxonomy_id))
                        if not os.path.exists(os.path.join(outdir, taxonomy_id+'_images')):
                            os.makedirs(os.path.join(outdir, taxonomy_id+'_images'))
                        # save pred, gt, partial pcds 
                        pred = pcds_pred[-1]
                        for mm, model_name in enumerate(model_id):
                            output_file = os.path.join(outdir, taxonomy_id, model_name+'_{:02d}'.format(partial_id))
                            write_ply(output_file + '_pred.ply', pred[mm, :].detach().cpu().numpy(), ['x', 'y', 'z'])
                            write_ply(output_file + '_gt.ply', gt[mm, :].detach().cpu().numpy(), ['x', 'y', 'z'])
                            write_ply(output_file + '_partial.ply', partial[mm, :].detach().cpu().numpy(), ['x', 'y', 'z'])
                            # output img files
                            img_filename = os.path.join(outdir, taxonomy_id+'_images', model_name+'.jpg')
                            output_img = pc_util.point_cloud_three_views(pred[mm, :].detach().cpu().numpy(), diameter=7)
                            output_img = (output_img*255).astype('uint8')
                            im = Image.fromarray(output_img)
                            im.save(img_filename)


        # Record category results
        self.train_record('============================ TEST RESULTS ============================')
        self.train_record('Taxonomy\t#Sample\t' + '\t'.join(test_metrics.items))

        for taxonomy_id in category_metrics:
            message = '{:s}\t{:d}\t'.format(taxonomy_id, category_metrics[taxonomy_id].count(0)) 
            message += '\t'.join(['%.4f' % value for value in category_metrics[taxonomy_id].avg()])
            mclass_metrics.update(category_metrics[taxonomy_id].avg())
            self.train_record(message)

        self.train_record('Overall\t{:d}\t'.format(test_metrics.count(0)) + '\t'.join(['%.4f' % value for value in test_metrics.avg()]))
        self.train_record('MeanClass\t\t' + '\t'.join(['%.4f' % value for value in mclass_metrics.avg()]))

        # record testing results
        message = '#{:d} {:.4f} {:.4f} {:.4f} {:.4f} | {:.4f} | #{:d} {:.4f}'.format(self.epoch, test_losses.avg(0), test_losses.avg(1), test_losses.avg(2), test_losses.avg(4), test_losses.avg(3), self.best_epoch, self.best_metrics)
        self.test_record(message)


        return test_losses.avg(3)
