import os
import logging
from datetime import datetime
import time
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch

from dataset import SoccerNetClips, SoccerNetClipsTesting, SoccerNetClassification
from model import Video2Spot, Video2Classifcation
from train import trainer, test_spotting
from loss import NLLLoss

import wandb

def main(args):

    logging.info("Parameters:")
    for arg in vars(args):
        logging.info(arg.rjust(15) + " : " + str(getattr(args, arg)))

    # create dataset
    if not args.test_only:
        dataset_Train = SoccerNetClassification(path=args.SoccerNet_path, features=args.features, split=args.split_train, version=args.version, framerate=args.framerate, window_size=args.window_size_caption)
        dataset_Valid = SoccerNetClassification(path=args.SoccerNet_path, features=args.features, split=args.split_valid, version=args.version, framerate=args.framerate, window_size=args.window_size_caption)
        dataset_Valid_metric  = SoccerNetClassification(path=args.SoccerNet_path, features=args.features, split=args.split_valid, version=args.version, framerate=args.framerate, window_size=args.window_size_caption)
    dataset_Test  = SoccerNetClassification(path=args.SoccerNet_path, features=args.features, split=args.split_test, version=args.version, framerate=args.framerate, window_size=args.window_size_caption)

    if args.feature_dim is None:
        args.feature_dim = dataset_Test[0][0].shape[-1]
        print("feature_dim found:", args.feature_dim)
    # create model
    feature_size = {"gpt2": 768, "gpt2-medium": 1024, "gpt2-large": 1280, "gpt2-xl": 1600}
    model = Video2Classifcation(num_classes=len(dataset_Test.class_labels), weights=args.load_weights, input_size=args.feature_dim,
                  window_size=args.window_size_spotting, 
                  vlad_k=args.vlad_k,
                  framerate=args.framerate, pool=args.pool, freeze_encoder=args.freeze_encoder, weights_encoder=args.weights_encoder,
                  proj_size=feature_size[args.gpt_type]).cuda()
    logging.info(model)
    total_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    parameters_per_layer  = [p.numel() for p in model.parameters() if p.requires_grad]
    logging.info("Total number of parameters: " + str(total_params))

    # create dataloader
    if not args.test_only:
        train_loader = torch.utils.data.DataLoader(dataset_Train,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.max_num_worker, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(dataset_Valid,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.max_num_worker, pin_memory=True)

        val_metric_loader = torch.utils.data.DataLoader(dataset_Valid_metric,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.max_num_worker, pin_memory=True)


    # training parameters
    if not args.test_only:
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.LR)

        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=args.patience)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        # start training
        trainer("classifying", train_loader, val_loader, val_metric_loader, 
                model, optimizer, scheduler, criterion,
                model_name=args.model_name,
                max_epochs=10, evaluation_frequency=args.evaluation_frequency)

    return 