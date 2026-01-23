import os
import numpy as np
import torch
import pandas as pd
from PIL import Image
from utils import *
from torchvision.utils import save_image
np.random.seed(22)  # for reproducibility

class Landmark_regression_dataset(torch.utils.data.Dataset):
    """
    Dataset that contains the landamrk regression task
    Given an X-ray image along the patient, we predict the distance needed to travel to each landmark across the body
    args:
        - root_data: path where the X-ray images are saved
        - root_data: path where .csv annotations are saved (for both landmark and full X-ray)
        - augmentation: bool, implement data augmentation
        - augmentation_params: dict, parameters for augmentation has keys: ['p', 'shift']
        - head_only: bool, set True if you want head data only
        - mode: choose ['train', 'calibration', 'test']
        - size: image size, e.g., [256, 256]
    """
    def __init__(self,
                 root_data='./data/',
                 augmentation=True,
                 augmentation_params={"p": 0.15, "shift": 0.15},
                 head_only=False,
                 mode='train',
                 size=[224,224]):
        
        super(Landmark_regression_dataset, self).__init__()
        
        self.root_data = root_data
        self.augmentation = augmentation
        self.augmentation_params = augmentation_params
        self.mode = mode
        self.size = size
        self.transform = transform(size=self.size, augmentation=self.augmentation)

        # process annotations (this is specific to the dataset structure)
        landmarks = pd.read_csv(f'{root_data}/landmark_dataset.csv', index_col=0)
        landmarks['landmark'] = landmarks['file_path'].str.split('/').str[1].astype(int)
        landmarks = landmarks.rename(columns={"normalized_x":"y", "normalized_y":"x", "normalized_z":"z"})
        landmarks["y"] = 1 - landmarks["y"]
        
        landmarks = landmarks[landmarks['split'] == self.mode]
        
        if head_only:
            landmarks = landmarks[landmarks['landmark'] == 1]
        else:
            landmarks = landmarks[landmarks['landmark'] <= 14]
        
        annotated_cases = landmarks['patient'].unique() # all cases regardeless of train or test
        self.target_locations = landmarks.groupby(['patient', 'landmark'])[['x', 'y', 'z']].mean() # mean of all residents annotating the same landmark
        
        self.annotations = pd.read_csv(f'{root_data}/gaussian_xyz_dataset.csv', index_col=0)
        self.annotations = self.annotations[self.annotations['patient'].isin(annotated_cases)]
        self.annotations = self.annotations.rename(columns={"normalized_x":"y", "normalized_y":"x", "normalized_z":"z"})
        self.annotations["y"] = 1 - self.annotations["y"]
        
        # landmark connectivity graph (will be used for skeleton loss)
        self.skeleton_pairs = [
            (0,9), (9,10), (1,9), (2,9), (10,13), (1,5), (2,6), (5,7), (6,8), 
            (3,11), (4,12), (11,13), (12, 13), (11,13)
            ]

        # ct size for each patient to normalize 0-1
        self.ct_size = pd.read_csv(f'{root_data}/ct_size.csv', index_col=0).set_index('patient')
        self.ct_size = self.ct_size[['width', 'height', 'depth']]

        if not head_only:
            full_landmarks = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
            
            multi_index = pd.MultiIndex.from_product([annotated_cases, full_landmarks], names=['patient', 'landmark'])
            self.target_locations = self.target_locations.reindex(multi_index).reset_index()
            
            self.target_locations[['x','y','z']] = self.target_locations[['x','y', 'z']].fillna(-99)
            self.target_locations = self.target_locations.set_index(['patient', 'landmark'])

            del full_landmarks, multi_index
        
        del landmarks, annotated_cases
        
    def __getitem__(self, index):
        sample = self.annotations.iloc[index]

        X_ray = Image.open(sample.file_path, 'r').convert('L')
        X_ray = self.transform(X_ray).repeat(3,1,1)

        x, y, z = sample[['x', 'y', 'z']].to_numpy(dtype=np.float32)
        landmark_locations = self.target_locations.loc[sample.patient]

        if self.augmentation:
            if np.random.uniform() < self.augmentation_params["p"]:
                # Randomly shift the X-ray image
                shift_x, shift_y = np.random.uniform(low=-self.augmentation_params["shift"], high=self.augmentation_params["shift"], size=2)
                x += shift_x
                y += shift_y
                landmark_locations[['x', 'y']] += [shift_x, shift_y]

                if (x < 0) or (x > 1) or (y < 0) or (y > 1):
                    #revert the shift if out of bounds
                    x -= shift_x
                    y -= shift_y
                    landmark_locations[['x', 'y']] -= [shift_x, shift_y]

        # flag out of bounds landmarks so they do not influence the loss
        # 0 means out of bounds, 1 means in bounds, so we can easility multiply the loss with this
        landmark_visibility = 1 - ((landmark_locations < 0).any(axis=1) | (landmark_locations > 1).any(axis=1)).to_numpy()

        target = torch.tensor((landmark_locations - [x,y,z]).to_numpy(dtype=np.float32).flatten())
        current_position = torch.tensor([x,y,z]).view(-1)
        ct_size = self.ct_size.loc[sample.patient].to_numpy() #[height, width, depth]
        
        if self.mode == 'test' or self.mode == 'calibration':
            return (X_ray, current_position, landmark_visibility, target, sample['patient'], ct_size)
        else:
            return (X_ray, current_position, landmark_visibility, target)

    def __len__(self):
        return len(self.annotations)


if __name__ == "__main__":
    # sanity check
    dataset = Landmark_regression_dataset(mode='train')
    X_ray, current_position, landmark_visibility, target = dataset[0]
    print("-"*10, " training dataset sample ", "-"*10)
    print("X_ray.shape: ", X_ray.shape)
    print("current_position.shape: ", current_position.shape)
    print("landmark_visibility.shape: ", landmark_visibility.shape)
    print("target.shape: ", target.shape)

    dataset = Landmark_regression_dataset(mode='calibration')
    X_ray, current_position, landmark_visibility, target, case_ID, ct_size = dataset[0]
    print("-"*10, " calibration dataset sample ", "-"*10)
    print("X_ray.shape: ", X_ray.shape)
    print("current_position.shape: ", current_position.shape)
    print("landmark_visibility.shape: ", landmark_visibility.shape)
    print("target.shape: ", target.shape)
    print("case_ID: ", case_ID)
    print("ct_size.shape: ", ct_size.shape)

    dataset = Landmark_regression_dataset(mode='test')
    X_ray, current_position, landmark_visibility, target, case_ID, ct_size = dataset[0]
    print("-"*10, " test dataset sample ", "-"*10)
    print("X_ray.shape: ", X_ray.shape)
    print("current_position.shape: ", current_position.shape)
    print("landmark_visibility.shape: ", landmark_visibility.shape)
    print("target.shape: ", target.shape)
    print("case_ID: ", case_ID)
    print("ct_size.shape: ", ct_size.shape)
    