import torch.nn as nn
import torch
from dataset import Landmark_regression_dataset
from models import *
import argparse
import os
import sys
from tqdm import tqdm
from utils import *

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training landmark regression model")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        help=("Be careful when using deep green or black diamond")
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        required=True,
        help=("the name of the directory in log_dir")
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default='resnet34',
        required=False,
        help=("the name of the backbone used in training choose [resnet34, resnet50, resnet101, convnext_tiny, convnext_base, convnext_large, vit_b_16, vit_l_16]")
    )
    parser.add_argument(
        "--dropout_prob",
        type=float,
        default=0.2
    )
    parser.add_argument(
        "--augmentation_strength",
        type=str,
        default='weak',
        required=False,
        help=("choose ['weak', 'mid', 'strong']. This will determine the strength of the patient movement augmentation applied to the training data.")
    )
    parser.add_argument(
        "--head_only",
        action='store_true'
    )
    parser.add_argument(
        "--is_probabilistic",
        action="store_true"
    )
    parser.add_argument(
        "--skeleton_distance_loss",
        action="store_true"
    )
    parser.add_argument(
        "--Lambda",
        type=float,
        default=1,
        help=("weight for the skeleton distance loss, only used if --skeleton_distance_loss is set")
    )
    parser.add_argument(
        "--Sigma",
        type=float,
        default=5,
        help=("weight for the variance regularization term, only used if --is_probabilistic is set")
    )

    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args

def main(args):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_count = torch.cuda.device_count()

    # logging stuff
    log_dir = f'{args.log_dir}/{args.exp_name}'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(f'{log_dir}/checkpoints', exist_ok=True)
    
    img_size = [224,224]

    if args.augmentation_strength == 'weak':
        augmentation_params = {'p': 0.2, 'shift': 0.05}
    elif args.augmentation_strength == 'mid':
        augmentation_params = {'p': 0.2, 'shift': 0.15}
    elif args.augmentation_strength == 'strong':
        augmentation_params = {'p': 0.2, 'shift': 0.25}
    elif args.augmentation_strength == 'none':
        augmentation_params = {'p': 0.0, 'shift': 0.0}

    dataset = Landmark_regression_dataset(augmentation=True, augmentation_params=augmentation_params, head_only=args.head_only, mode='train', size=img_size)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    
    if args.is_probabilistic:
        if args.head_only:
            outs = 1*3*2
        else:
            outs = 14*3*2
    else:
        if args.head_only:
            outs = 1*3*1
        else:
            outs = 14*3*1
            
    if gpu_count>1:
        model = nn.DataParallel(Landmark_regression_model(backbone=args.model_name, outs=outs, positional_encoding=args.positional_encoding, p=args.dropout_prob))
        model = model.to(DEVICE).to(torch.float32)
    else:
        model = Landmark_regression_model(backbone=args.model_name, outs=outs, positional_encoding=args.positional_encoding, p=args.dropout_prob).to(DEVICE).to(torch.float32)
        
    loss_MSE = nn.MSELoss()
    loss_probabilistic = nn.GaussianNLLLoss(reduction='none')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    Lambda_ = args.Lambda
    Sigma_ = args.Sigma

    print(f'Training on {DEVICE} will start, using {gpu_count} GPUS!')
    print(vars(args))
    print('-'*100)

    loss_list = []
    loss_nll_list = []
    loss_skeleton_distance_list = []
    variance_list = []
    
    for e in range(args.epochs):
        model.train()

        for i, batch in enumerate(tqdm(loader)):
            X_ray, current_position, landmark_visibility, needed_displacement_gt = batch
            
            X_ray = X_ray.to(DEVICE).to(torch.float32)
            landmark_visibility = landmark_visibility.to(DEVICE).to(torch.float32)
            needed_displacement_gt = needed_displacement_gt.to(DEVICE).to(torch.float32)

            if args.positional_encoding:
                current_position = current_position.to(DEVICE).to(torch.float32)
                pred = model(X_ray, current_position)
            else:
                pred = model(X_ray)
            
            if args.is_probabilistic:
                means = pred[0]  # (B, outs//2)
                variance = pred[1]
                
                penalty = Sigma_*variance.mean()  # regularization term to prevent variance from going to zero
                loss_nll = loss_probabilistic(means, needed_displacement_gt, variance) # (B, outs//2)
                
                landmark_visibility_flatten = landmark_visibility.unsqueeze(-1).repeat(1, 1, 3).reshape(landmark_visibility.shape[0], -1) # (B, outs//2)
                loss_ = loss_nll * landmark_visibility_flatten # mask out the landmarks that are not visible
                loss_ = loss_.sum() / landmark_visibility_flatten.sum().clamp(min=1.) # reduce loss over visible landmarks
                
                loss_ += penalty  # add regularization term
                
                if args.skeleton_distance_loss:
                    loss_skeleton_distance = skeleton_distance_loss(means, needed_displacement_gt, landmark_visibility, skeleton_pairs=dataset.skeleton_pairs) # (B, outs//2)
                    loss_ += Lambda_*loss_skeleton_distance
                    
            else:
                loss_ = loss_MSE(pred, needed_displacement_gt)
            
            if torch.isnan(loss_):
                print('Loss is NaN, DANGER! will skip this batch')
                continue

            loss_.backward()

            optimizer.step()
            optimizer.zero_grad()

            print(f'loss: {loss_}')
            sys.stdout.flush()

            if i%50 == 0:
                loss_list.append(loss_.item())
                loss_skeleton_distance_list.append(loss_skeleton_distance.item() if args.skeleton_distance_loss else 0.0)
                loss_nll_list.append(loss_nll.mean().item() if args.is_probabilistic else 0.0)
                variance_list.append(variance.mean().item() if args.is_probabilistic else 0.0)
        
        print(f'finished epoch {e}\n----------------------------------------------------')
        sys.stdout.flush()

        torch.save({
                'model_state_dict': model.state_dict(),
                'train_loss': loss_list,
                'train_loss_nll': loss_nll_list,
                'train_loss_skeleton_distance': loss_skeleton_distance_list,
                'variance': variance_list,
                'epoch': e,
                'args': vars(args),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'{log_dir}/checkpoints/chkpt_e{e}.pt')

        torch.cuda.empty_cache()

if __name__=="__main__":
    args = parse_args()
    main(args)