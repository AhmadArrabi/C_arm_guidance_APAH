import torch.nn as nn
import torch
from dataset import Landmark_regression_dataset
from models import *
import argparse
import os
from utils import *
import sys
from tqdm import tqdm
import numpy as np
from torchvision.utils import save_image
import os

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training landmark regression model")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
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
        "--head_only",
        action='store_true'
    )
    parser.add_argument(
        "--is_probabilistic",
        action="store_true"
    )
    parser.add_argument(
        "--positional_encoding",
        action="store_true"
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args

def main(args):
    print(f'calculating the nonconformity scores for landmark regression\nThe intervals will be saved in {args.log_dir}/{args.exp_name}')
    print(vars(args))
    print('-'*100)
    log_dir = f"{args.log_dir}/{args.exp_name}"
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    img_size = [224, 224]

    dataset_val = Landmark_regression_dataset(augmentation=False, mode='calibration', head_only=args.head_only, size=img_size)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    
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
    
    model = Landmark_regression_model(backbone=args.model_name, outs=outs, positional_encoding=args.positional_encoding, p=args.dropout_prob).to(DEVICE).to(torch.float32)

    # load most recent checkpoint
    checkpoint_dir = f'{log_dir}/checkpoints'
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
    most_recent_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[0]) if checkpoint_files else None

    if most_recent_checkpoint:
        checkpoint = torch.load(most_recent_checkpoint, map_location=DEVICE)
        state_dict = checkpoint['model_state_dict']
            
        model.load_state_dict(state_dict)
    
        print(f'loaded {most_recent_checkpoint} weights !!')
        sys.stdout.flush()
    else:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

    nll_list = []
    nonconformity_scores = []
    nonconformity_scores_mm = []
    euclidean_distances_mm = []
    euclidean_distances = []
    landmark_visibility_list = []

    loss_probabilistic = nn.GaussianNLLLoss(reduction='none')
    
    model.eval()
    enable_dropout(model)
    alphas = [0.9, 0.95, 0.97]
    eval_dict = {}
    T = 20 # number of MC dropout samples

    with torch.no_grad():    
        for step, batch in  tqdm(enumerate(loader_val)):
            X_ray, current_position, landmark_visibility, needed_displacement_gt, patient, ct_size = batch

            X_ray = X_ray.to(DEVICE).to(torch.float32)
            landmark_visibility = landmark_visibility.to(DEVICE).to(torch.float32)
            needed_displacement_gt = needed_displacement_gt.to(DEVICE).to(torch.float32)
            ct_size = ct_size.to(DEVICE).to(torch.float32)

            if args.positional_encoding:
                current_position = current_position.to(DEVICE).to(torch.float32)
                
            preds = []
            aleatoric_vars = []
            for _ in range(T):
                if args.positional_encoding:
                    output = model(X_ray, current_position)
                else:
                    output = model(X_ray)
                
                if args.is_probabilistic:
                    mean_pred, var_pred = output
                    preds.append(mean_pred)
                    aleatoric_vars.append(var_pred)
                else:
                    preds.append(output)
            
            preds = torch.stack(preds)
            mean_pred = preds.mean(0)  
            epistemic_var = preds.var(0)
            
            if args.is_probabilistic:
                aleatoric_vars = torch.stack(aleatoric_vars)
                aleatoric_vars = aleatoric_vars.mean(0)  # average over MC samples             
                
            if args.is_probabilistic:
                total_var = epistemic_var + aleatoric_vars
                metrics = calculate_metrics(mean_pred, total_var, needed_displacement_gt, landmark_visibility, ct_size, loss_probabilistic)
                nll_list.append(metrics['loss_nll'])
                nonconformity_scores_mm.append(metrics['nonconformity_score_mm'])
                landmark_visibility_list.append(landmark_visibility.cpu().detach().numpy())
                
            else:
                raise NotImplementedError("Non-probabilistic model is not supported for calibration interval generation")

        eval_nll = np.stack(nll_list, axis=0).mean()
        eval_nonconformity_scores_mm = np.stack(nonconformity_scores_mm, axis=0).reshape(-1, 14)
        eval_landmark_visibility = np.stack(landmark_visibility_list, axis=0).reshape(-1, 14)

        valid_cases = eval_landmark_visibility.sum(axis=0)

        for alpha in alphas:
            eval_dict[f'nonconformity_mm_{alpha}_quantile'] = np.apply_along_axis(lambda col: np.quantile(col[col != 0], alpha), axis=0, arr=eval_nonconformity_scores_mm)
            
        eval_dict['nll'] = eval_nll
        
        torch.save(eval_dict, f'{log_dir}/calibration_stats_{DEVICE}.pt')        

if __name__=="__main__":
    args = parse_args()
    main(args)