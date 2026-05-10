'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''

###########################################################################
#  This script implements the training loop for a captioning model that uses
#  a confidence MLP to weight the loss for each caption during training. 
# The script includes functions for computing caption weights based on features
#  extracted from the captions, training the model with weighted captions, 
# and evaluating the model on validation and test datasets. The training loop
#  iterates over epochs, where in each epoch, the model is trained on the 
# training dataset, evaluated on the validation and test datasets, and 
# checkpoints are saved based on performance metrics. 
# The compute_caption_weights function is a crucial part of the training 
# process, as it allows the model to focus more on captions that are deemed 
# more confident by the MLP, while still learning from all available captions. 
# The confidence MLP is trained jointly with the main captioning model, and 
# its parameters are updated based on the loss computed from the weighted captions, 
# allowing it to learn to assign higher confidence scores to captions that 
# lead to better performance of the captioning model, thus improving the 
# overall training process.
########################################################################### 

import argparse
import os
try:
    import ruamel_yaml as yaml
except ImportError:
    from ruamel import yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from data.coco_weighted_caption_dataset import WeightedCaptionTrainDataset
from models.confidence_mlp import ConfidenceMLP

from models.blip import blip_decoder
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader
from data.utils import save_result, coco_caption_eval

def weighted_collate_fn(batch):
    return batch

#################################################### Compute caption weights using the confidence MLP and the provided features
###### The function takes in the features, confidence MLP, and configuration to compute the weights for each caption.
###### It applies a softmax function to the raw scores obtained from the confidence MLP, optionally applies weight clipping, and calculates the entropy of the weights.
###### The function returns the computed weights and the entropy value. 
###### The weights are used to weight the loss for each caption during training, allowing the model to focus more on captions that are deemed more confident by the MLP.
###### The entropy value can be used as a regularization term to encourage the model to assign more uniform weights across captions, preventing it from focusing too much on a single caption.
###### The confidence MLP is a small neural network that takes in the features and outputs a confidence score for each caption. The softmax function is used to convert these scores into probabilities (weights) that sum to 1. The temperature parameter can be used to control the sharpness of the distribution of weights, and the weight clipping can be used to prevent any single caption from dominating the loss.
###### The compute_caption_weights function is a crucial part of the training process, as it allows the model to learn from multiple captions for each image while giving more importance to those that are more likely to be correct, as determined by the confidence MLP.
###### The function also includes an option for weight clipping, which can help to prevent the model from assigning too much weight to any single caption, thus encouraging it to learn from a more diverse set of captions. The entropy of the weights is calculated as a measure of how spread out the weights are across the captions, with higher entropy indicating a more uniform distribution of weights.
###### The compute_caption_weights function is called during the training loop for each image and its associated captions, and the computed weights are used to weight the loss for each caption when backpropagating the gradients. This allows the model to focus more on captions that are deemed more confident by the MLP, while still learning from all available captions.
###### The confidence MLP is trained jointly with the main captioning model, and its parameters are updated based on the loss computed from the weighted captions. This allows the MLP to learn to assign higher confidence scores to captions that lead to better performance of the captioning model, thus improving the overall training process. 
###### The compute_caption_weights function is an essential component of the training process for the captioning model, as it enables the model to effectively utilize multiple captions for each image while giving more importance to those that are more likely to be correct, as determined by the confidence MLP. This can lead to improved performance of the captioning model, as it can learn from a more diverse set of captions while still focusing on those that are most relevant for each image.
###### The confidence MLP is a small neural network that takes in the features extracted from the image and outputs a confidence score for each caption. The softmax function is applied to these scores to convert them into probabilities (weights) that sum to 1. The temperature parameter can be used to control the sharpness of the distribution of weights, with higher temperatures leading to a more uniform distribution and lower temperatures leading to a more peaked distribution. The weight clipping option can be used to prevent any single caption from dominating the loss, which can help to encourage the model to learn from a more diverse set of captions. The entropy of the weights is calculated as a measure of how spread out the weights are across the captions, with higher entropy indicating a more uniform distribution of weights. This can be used as a regularization term during training to encourage the model to assign more uniform weights across captions, preventing it from focusing too much on a single caption.
####################################################
def compute_caption_weights(features, confidence_mlp, config):
    raw_scores = confidence_mlp(features).squeeze(-1)

    tau = float(config.get("temperature", 1.0))
    weights = torch.softmax(raw_scores / tau, dim=0)

    # Optionally apply weight clipping to prevent any single caption from dominating the loss. 
    # This can help to encourage the model to learn from a more diverse set of captions.
    if bool(config.get("use_weight_clipping", False)):
        clip_factor = float(config.get("clip_factor", 0.5))
        thr = (1.0 / len(weights)) * clip_factor
        keep = (weights > thr).float()

        if keep.sum().item() == 0:
            keep[weights.argmax()] = 1.0

        weights = weights * keep
        weights = weights / weights.sum().clamp_min(1e-8)

    entropy = -(weights * weights.clamp_min(1e-8).log()).sum()
    return weights, entropy

    features = features.to(device)
    raw_scores = confidence_mlp(features).squeeze(-1)

    tau = float(config.get("temperature", 1.0))
    weights = torch.softmax(raw_scores / tau, dim=0)

    if bool(config.get("use_weight_clipping", False)):
        clip_factor = float(config.get("clip_factor", 0.5))
        thr = (1.0 / len(captions)) * clip_factor
        keep = (weights > thr).float()

        if keep.sum().item() == 0:
            keep[weights.argmax()] = 1.0

        weights = weights * keep
        weights = weights / weights.sum().clamp_min(1e-8)

    per_caption_losses = []
    for cap in captions:
        loss_i = model(image, [cap])
        per_caption_losses.append(loss_i)

    per_caption_losses = torch.stack(per_caption_losses)
    loss = (weights * per_caption_losses).sum()

    entropy = -(weights * weights.clamp_min(1e-8).log()).sum()

    return loss, weights.detach(), per_caption_losses.detach(), entropy.detach()

def train(model, confidence_mlp, data_loader, optimizer, epoch, device, config):
    model.train()
    confidence_mlp.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("loss", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("entropy", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("avg_weight_max", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))

    header = f"Train Caption Weighted Epoch: [{epoch}]"
    print_freq = 50

    # Iterate over the training data loader, where each sample contains an image and its associated captions and features. For each sample, the image is moved to the appropriate device (GPU or CPU), and the captions and features are extracted. The optimizer's gradients are zeroed out before computing the caption weights using the compute_caption_weights function, which takes in the features, confidence MLP, and configuration to compute the weights for each caption. The loss for each caption is computed using the model, and then weighted by the computed weights before backpropagating the gradients. The total loss value for the sample is accumulated and logged using the metric logger, along with the learning rate, entropy of the weights, and average maximum weight across captions. After iterating through all samples in the data loader, the metric logger synchronizes between processes (if distributed training is used) and prints the averaged statistics for the epoch.
    for sample in metric_logger.log_every(data_loader, print_freq, header):
        sample = sample[0]
        image = sample["image"].unsqueeze(0).to(device)
        captions = sample["captions"]
        features = sample["features"].to(device)

        optimizer.zero_grad()

        weights, entropy = compute_caption_weights(
            features=features,
            confidence_mlp=confidence_mlp,
            config=config,
        )

        total_loss_value = 0.0
        # Compute the loss for each caption, weight it by the computed weights, and backpropagate the gradients. 
        # The total loss value for the sample is accumulated and logged using the metric logger, along with the learning rate, entropy of the weights, and average maximum weight across captions.
        for i, cap in enumerate(captions):
            loss_i = model(image, [cap])
            weighted_loss_i = weights[i] * loss_i

            weighted_loss_i.backward(retain_graph=(i < len(captions) - 1))

            total_loss_value += weighted_loss_i.item()

        optimizer.step()

        metric_logger.update(loss=total_loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(entropy=entropy.item())
        metric_logger.update(avg_weight_max=weights.max().item())

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}
@torch.no_grad()
def evaluate(model, data_loader, device, config):
    # evaluate
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Caption generation:'
    print_freq = 10

    result = []
    for image, image_id in metric_logger.log_every(data_loader, print_freq, header): 
        
        image = image.to(device)       
        
        captions = model.generate(image, sample=False, num_beams=config['num_beams'], max_length=config['max_length'], 
                                  min_length=config['min_length'])
        
        for caption, img_id in zip(captions, image_id):
            result.append({"image_id": img_id.item(), "caption": caption})
  
    return result


def main(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    print("Creating captioning dataset")
    _, val_dataset, test_dataset = create_dataset('caption_coco', config)

    train_dataset = WeightedCaptionTrainDataset(
        ann_path=config["weighted_train_json"],
        image_size=config["image_size"],
        prompt=config.get("prompt", ""),
        max_candidates=int(config.get("max_candidates", 3)),
    )

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(
            [train_dataset, val_dataset, test_dataset],
            [True, False, False],
            num_tasks,
            global_rank,
        )
    else:
        samplers = [None, None, None]

    train_loader, val_loader, test_loader = create_loader(
        [train_dataset, val_dataset, test_dataset],
        samplers,
        batch_size=[1, config["batch_size"], config["batch_size"]],
        num_workers=[int(config.get("train_num_workers", 4)), 4, 4],
        is_trains=[True, False, False],
        collate_fns=[weighted_collate_fn, None, None],
    )

    #############
    ## Create the BLIP captioning model and the confidence MLP, and set up the optimizer for training. The BLIP model is initialized with the specified configuration, and both the BLIP model and the confidence MLP are moved to the appropriate device (GPU or CPU). If distributed training is enabled, both models are wrapped in DistributedDataParallel for synchronized training across multiple processes. The optimizer is set up to optimize the parameters of both the BLIP model and the confidence MLP, with potentially different learning rates for each. The training loop then iterates over epochs, where in each epoch, the model is trained on the training dataset, evaluated on the validation and test datasets, and checkpoints are saved based on performance metrics.
    ## The training loop also includes logging of training and evaluation metrics, and handles distributed synchronization as needed. The compute_caption_weights function is called during the training loop to compute the weights for each caption based on the features and the confidence MLP, which are then used to weight the loss for each caption when backpropagating the gradients. This allows the model to focus more on captions that are deemed more confident by the MLP, while still learning from all available captions.
    ## The confidence MLP is trained jointly with the main captioning model, and its parameters are updated based on the loss computed from the weighted captions. This allows the MLP to learn to assign higher confidence scores to captions that lead to better performance of the captioning model, thus improving the overall training process.
    #############
    print("Creating model")
    model = blip_decoder(
        pretrained=config["pretrained"],
        image_size=config["image_size"],
        vit=config["vit"],
        vit_grad_ckpt=config["vit_grad_ckpt"],
        vit_ckpt_layer=config["vit_ckpt_layer"],
        prompt=config["prompt"],
    )
    model = model.to(device)

    confidence_mlp = ConfidenceMLP(
        in_dim=3,
        hidden_dim=int(config.get("mlp_hidden_dim", 64)),
        dropout=float(config.get("mlp_dropout", 0.1)),
    ).to(device)

    model_without_ddp = model
    confidence_mlp_without_ddp = confidence_mlp

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        confidence_mlp = torch.nn.parallel.DistributedDataParallel(
            confidence_mlp, device_ids=[args.gpu]
        )
        model_without_ddp = model.module
        confidence_mlp_without_ddp = confidence_mlp.module

    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": config["init_lr"]},
            {"params": confidence_mlp.parameters(), "lr": float(config.get("mlp_lr", config["init_lr"]))},
        ],
        weight_decay=config["weight_decay"],
    )

    best = 0
    best_epoch = 0

    print("Start training")
    start_time = time.time()

    for epoch in range(0, config["max_epoch"]):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            cosine_lr_schedule(optimizer, epoch, config["max_epoch"], config["init_lr"], config["min_lr"])

            train_stats = train(
                model,
                confidence_mlp,
                train_loader,
                optimizer,
                epoch,
                device,
                config,
            )

        val_result = evaluate(model_without_ddp, val_loader, device, config)
        val_result_file = save_result(val_result, args.result_dir, f"val_epoch{epoch}", remove_duplicate="image_id")

        test_result = evaluate(model_without_ddp, test_loader, device, config)
        test_result_file = save_result(test_result, args.result_dir, f"test_epoch{epoch}", remove_duplicate="image_id")

        if utils.is_main_process():
            coco_val = coco_caption_eval(config["coco_gt_root"], val_result_file, "val")
            coco_test = coco_caption_eval(config["coco_gt_root"], test_result_file, "test")

            if args.evaluate:
                log_stats = {
                    **{f"val_{k}": v for k, v in coco_val.eval.items()},
                    **{f"test_{k}": v for k, v in coco_test.eval.items()},
                }
                with open(os.path.join(args.output_dir, "evaluate.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")
            else:
                save_obj = {
                    "model": model_without_ddp.state_dict(),
                    "confidence_mlp": confidence_mlp_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                    "epoch": epoch,
                }

                if coco_val.eval["CIDEr"] + coco_val.eval["Bleu_4"] > best:
                    best = coco_val.eval["CIDEr"] + coco_val.eval["Bleu_4"]
                    best_epoch = epoch
                    torch.save(save_obj, os.path.join(args.output_dir, "checkpoint_best.pth"))

                torch.save(save_obj, os.path.join(args.output_dir, "checkpoint_last.pth"))

                log_stats = {
                    **{f"train_{k}": v for k, v in train_stats.items()},
                    **{f"val_{k}": v for k, v in coco_val.eval.items()},
                    **{f"test_{k}": v for k, v in coco_test.eval.items()},
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                }
                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

        if args.evaluate:
            break

        if args.distributed:
            dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))   


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/caption_weighted_subset.yaml')
    parser.add_argument('--output_dir', default='output/Caption_coco')        
    parser.add_argument('--evaluate', action='store_true')    
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)