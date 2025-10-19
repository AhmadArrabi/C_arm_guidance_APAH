from torchvision import transforms
import pandas as pd
import torch.nn as nn
import SimpleITK as sitk
import numpy as np
import torch
import matplotlib.pyplot as plt

def calculate_test_metrics(means, variance, needed_displacement_gt, landmark_visibility, ct_size, loss_probabilistic, eval_dict):
    # nll is a number
    # residuals are lists of 14
    loss_nll = loss_probabilistic(means, needed_displacement_gt, variance) # (B, 42)
    landmark_visibility_flatten = landmark_visibility.unsqueeze(-1).repeat(1, 1, 3).reshape(landmark_visibility.shape[0], -1) # (B, 42)
    
    loss_nll = loss_nll * landmark_visibility_flatten # mask out the landmarks that are not visible
    loss_nll = loss_nll.sum() / landmark_visibility_flatten.sum().clamp(min=1.) # reduce loss over visible landmarks
    
    # raw normalized residuals (these will be used for the calibration interval)
    means = means.reshape(means.shape[0], -1, 3)  
    variance = variance.reshape(variance.shape[0], -1, 3)  
    needed_displacement_gt = needed_displacement_gt.reshape(needed_displacement_gt.shape[0], -1, 3)
    
    euclidean_distances = torch.norm(needed_displacement_gt - means, dim=-1) * landmark_visibility # (B,14)
    
    within_range_flag_90_euclidean = (euclidean_distances <= torch.from_numpy(eval_dict['euclidean_0.9_quantile']).to(euclidean_distances.device)) * landmark_visibility 
    within_range_flag_95_euclidean = (euclidean_distances <= torch.from_numpy(eval_dict['euclidean_0.95_quantile']).to(euclidean_distances.device)) * landmark_visibility 
    within_range_flag_97_euclidean = (euclidean_distances <= torch.from_numpy(eval_dict['euclidean_0.97_quantile']).to(euclidean_distances.device)) * landmark_visibility 

    within_range_flag_90_nonconformity = (euclidean_distances <= (torch.from_numpy(eval_dict['nonconformity_0.9_quantile']).to(euclidean_distances.device)*(variance.sum(dim=2)+1e-06))) * landmark_visibility 
    within_range_flag_95_nonconformity = (euclidean_distances <= (torch.from_numpy(eval_dict['nonconformity_0.95_quantile']).to(euclidean_distances.device)*(variance.sum(dim=2)+1e-06))) * landmark_visibility 
    within_range_flag_97_nonconformity = (euclidean_distances <= (torch.from_numpy(eval_dict['nonconformity_0.97_quantile']).to(euclidean_distances.device)*(variance.sum(dim=2)+1e-06))) * landmark_visibility 

    # mm residuals 0-1 -> mm (for interprations and paper writing)
    ct_size = ct_size.unsqueeze(1).repeat(1, 14, 1)
    means_mm = means * ct_size 
    variance_mm = variance * ct_size
    needed_displacement_gt_mm = needed_displacement_gt * ct_size

    euclidean_distances_mm = torch.norm(needed_displacement_gt_mm - means_mm, dim=-1) * landmark_visibility # (B,14)
    
    within_range_flag_90_euclidean_mm = (euclidean_distances_mm <= torch.from_numpy(eval_dict['euclidean_mm_0.9_quantile']).to(euclidean_distances_mm.device)) * landmark_visibility 
    within_range_flag_95_euclidean_mm = (euclidean_distances_mm <= torch.from_numpy(eval_dict['euclidean_mm_0.95_quantile']).to(euclidean_distances_mm.device)) * landmark_visibility 
    within_range_flag_97_euclidean_mm = (euclidean_distances_mm <= torch.from_numpy(eval_dict['euclidean_mm_0.97_quantile']).to(euclidean_distances_mm.device)) * landmark_visibility 

    print('variance-'*80)
    print(variance_mm)
    print('actuall boundry-'*80)
    print(torch.from_numpy(eval_dict['nonconformity_mm_0.9_quantile']).to(euclidean_distances.device)*(variance_mm.sum(dim=2)))
    within_range_flag_90_nonconformity_mm = (euclidean_distances_mm <= (torch.from_numpy(eval_dict['nonconformity_mm_0.9_quantile']).to(euclidean_distances.device)*(variance_mm.sum(dim=2)+1e-06))) * landmark_visibility 
    within_range_flag_95_nonconformity_mm = (euclidean_distances_mm <= (torch.from_numpy(eval_dict['nonconformity_mm_0.95_quantile']).to(euclidean_distances.device)*(variance_mm.sum(dim=2)+1e-06))) * landmark_visibility 
    within_range_flag_97_nonconformity_mm = (euclidean_distances_mm <= (torch.from_numpy(eval_dict['nonconformity_mm_0.97_quantile']).to(euclidean_distances.device)*(variance_mm.sum(dim=2)+1e-06))) * landmark_visibility 

    return {'loss_nll': loss_nll.item(),
            'euclidean_distances_mm': euclidean_distances_mm.cpu().detach().numpy(),
            'within_range_flag_90_euclidean': within_range_flag_90_euclidean.cpu().detach().numpy(),
            'within_range_flag_95_euclidean': within_range_flag_95_euclidean.cpu().detach().numpy(),
            'within_range_flag_97_euclidean': within_range_flag_97_euclidean.cpu().detach().numpy(),
            'within_range_flag_90_nonconformity': within_range_flag_90_nonconformity.cpu().detach().numpy(),
            'within_range_flag_95_nonconformity': within_range_flag_95_nonconformity.cpu().detach().numpy(),
            'within_range_flag_97_nonconformity': within_range_flag_97_nonconformity.cpu().detach().numpy(),
            'within_range_flag_90_euclidean_mm': within_range_flag_90_euclidean_mm.cpu().detach().numpy(),
            'within_range_flag_95_euclidean_mm': within_range_flag_95_euclidean_mm.cpu().detach().numpy(),
            'within_range_flag_97_euclidean_mm': within_range_flag_97_euclidean_mm.cpu().detach().numpy(),
            'within_range_flag_90_nonconformity_mm': within_range_flag_90_nonconformity_mm.cpu().detach().numpy(),
            'within_range_flag_95_nonconformity_mm': within_range_flag_95_nonconformity_mm.cpu().detach().numpy(),
            'within_range_flag_97_nonconformity_mm':within_range_flag_97_nonconformity_mm.cpu().detach().numpy()
            }

def calculate_metrics(means, variance, needed_displacement_gt, landmark_visibility, ct_size, loss_probabilistic):
    # nll is a number
    # residuals are lists of 14
    loss_nll = loss_probabilistic(means, needed_displacement_gt, variance) # (B, 42)
    landmark_visibility_flatten = landmark_visibility.unsqueeze(-1).repeat(1, 1, 3).reshape(landmark_visibility.shape[0], -1) # (B, 42)
    
    loss_nll = loss_nll * landmark_visibility_flatten # mask out the landmarks that are not visible
    loss_nll = loss_nll.sum() / landmark_visibility_flatten.sum().clamp(min=1.) # reduce loss over visible landmarks
    
    # raw normalized residuals (these will be used for the calibration interval)
    means = means.reshape(means.shape[0], -1, 3)  
    variance = variance.reshape(variance.shape[0], -1, 3)  
    needed_displacement_gt = needed_displacement_gt.reshape(needed_displacement_gt.shape[0], -1, 3)
    
    # mm residuals 0-1 -> mm 
    ct_size = ct_size.unsqueeze(1).repeat(1, 14, 1)
    means_mm = means * ct_size 
    variance_mm = variance * ct_size
    needed_displacement_gt_mm = needed_displacement_gt * ct_size

    euclidean_distances_mm = torch.norm(needed_displacement_gt_mm - means_mm, dim=-1) * landmark_visibility
    nonconformity_score_mm = euclidean_distances_mm / (variance_mm.sum(dim=2) + 1e-06)
    
    return {'loss_nll': loss_nll.item(),
            'nonconformity_score_mm': nonconformity_score_mm.cpu().detach().numpy()}

def transform(size, augmentation=True):
    if augmentation:
        return transforms.Compose([
            transforms.Resize(size=tuple(size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(size=tuple(size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
def get_normalizing_constants(root='/gpfs2/scratch/aarrabi/Carm_positioning/data'):
    landmarks = pd.read_csv(f'{root}/full_annotations_2.csv', index_col=0)
    landmarks = landmarks[~landmarks['filename'].str.contains('NONE')]
    annotated_cases = landmarks.case_number.unique()
    
    annotations = pd.read_csv(f'{root}/full_dataset.csv', index_col=0)
    annotations = annotations[annotations.case_number.isin(annotated_cases)]
    train_set = annotations[annotations['mode'] == 'train']

    return [train_set['x_modified'].max(), train_set['y_modified'].max(), train_set['z_modified'].max()]

def denormalize_label(label):
    normalizing_constants = get_normalizing_constants()
    if label.shape[-1] == 40:
        label[::2] = label[::2]*normalizing_constants[0]    # X
        label[1::2] = label[1::2]*normalizing_constants[1]  # Y
    elif label.shape[-1] == 2:
        label[:,0] = label[:,0]*normalizing_constants[0]    # X
        label[:,1] = label[:,1]*normalizing_constants[1]    # Y
    else:
        raise IndexError(f'Denormalized position needs to be of shape 40 for multiple landmarks and 2 for one')
    
    return label

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
        #if module.__class__.__name__.startswith('Dropout'):
            module.train()

def change_dropout(model, p):
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.p = p  # e.g., new_p = 0.1

def load_nifti_with_metadata(file_path):
    image = sitk.ReadImage(file_path)
    return image

def resample_to_isotropic(image, new_spacing=(1.0, 1.0, 1.0)):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    
    # Compute new size to preserve physical dimensions
    new_size = [
        int(round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampled_image = resampler.Execute(image)

    return resampled_image, new_size

def compute_mip(image, axis=0):
    mip = sitk.MaximumProjection(image, axis)
    return sitk.GetArrayFromImage(mip)

def normalize(img):
    img = img - np.min(img)
    img = img / (np.max(img) + 1e-5)
    return (img * 255).astype(np.uint8)

def map_coords_back(resampled_coords, original_spacing, new_spacing):
    zoom_factors = np.array(original_spacing) / np.array(new_spacing)
    original_coords = np.array(resampled_coords) / zoom_factors
    return original_coords

def revert_isotropic(ct_root = "/netfiles/vaillab/upper_nifti",
                     root_annotations = "/gpfs2/scratch/aarrabi/Carm_positioning/data"
                     ):
    
    landmarks = pd.read_csv(f'{root_annotations}/full_annotations_2.csv', index_col=0)
    landmarks = landmarks[(landmarks['part'] == "upper") & (landmarks['landmark'] == 0)]
    rows = []
    
    for case_id in landmarks['case_number'].unique():
        coordinates = landmarks[landmarks['case_number'] == case_id][["z_modified", "x_modified", "y_modified"]]
        nifti = load_nifti_with_metadata(f"{ct_root}/{case_id}_BONE_H-N-UXT_3X3.nii.gz")
        nifti, new_size = resample_to_isotropic(nifti, new_spacing=(1,1,1))

        rows.append({"case_number": case_id, 
                     "x": coordinates["y_modified"].values.tolist()[0] / new_size[0],
                     "y": coordinates["x_modified"].values.tolist()[0] / new_size[2],
                     "z": coordinates["z_modified"].values.tolist()[0] / new_size[1]})
        print(case_id)
    
    df = pd.DataFrame(rows)
    df.to_csv("/gpfs2/scratch/aarrabi/Carm_positioning/data/head_ct_annotations.csv")

def skeleton_distance_loss(output, target, visibility, skeleton_pairs):
    """
    Args:
        output: (B, 84) - predicted means + variances
        target: (B, 42) - ground-truth means
        visibility: (B, 14) - binary mask for visible landmarks
        skeleton_pairs: list of (i, j) pairs
    Returns:
        scalar loss (averaged over valid pairs)
    """
    B = output.shape[0]

    mean_pred = output.view(B, 14, 3)  # (B, 14, 3)
    mean_gt   = target.view(B, 14, 3)

    pred_xy = mean_pred[:, :, :2]
    gt_xy   = mean_gt[:, :, :2]

    total_loss = 0.0
    total_valid = 0

    for i, j in skeleton_pairs:
        vis_mask = visibility[:, i] * visibility[:, j]  # (B,) - 1 if both visible

        if vis_mask.sum() == 0:
            continue  # skip if no valid pair in batch

        d_pred = torch.norm(pred_xy[:, i] - pred_xy[:, j], dim=1)
        d_gt   = torch.norm(gt_xy[:, i] - gt_xy[:, j], dim=1)
        d_diff = (d_pred - d_gt) ** 2

        masked_loss = d_diff * vis_mask  # only valid samples
        total_loss += masked_loss.sum()
        total_valid += vis_mask.sum()

    return total_loss / (total_valid + 1e-6)


def plot_landmark_pose(landmarks=None, skeleton=None, title="Pose"):
    """
    Args:
        landmarks: tensor (N, 2) in [0, 1] (x, y)
        skeleton: list of (i, j) landmark pairs
    """

    N = landmarks.shape[0]

    plt.figure(figsize=(4, 4))
    plt.xlim(0, 1)
    plt.ylim(1, 0)

    landmarks = landmarks.detach().cpu().numpy()
    print(landmarks)
    
    # Draw skeleton
    if skeleton:
        for i, j in skeleton:
            x = [landmarks[i, 0], landmarks[j, 0]]
            y = [landmarks[i, 1], landmarks[j, 1]]
            plt.plot(x, y, linewidth=4)

    # Draw landmarks
    for idx, (x, y) in enumerate(landmarks):
        plt.plot(x, y, 'o',color='black', markersize=7)  # visible = green
        
    plt.title(title)
    plt.axis('off')
    plt.gca().invert_yaxis()
    plt.savefig("SKELETON.svg", bbox_inches='tight', pad_inches=0.1)



