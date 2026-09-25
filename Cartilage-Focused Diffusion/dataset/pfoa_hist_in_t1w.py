from torch.utils.data import Dataset
import torchio as tio
import os

PREPROCESSING_TRANSORMS = tio.Compose([
    tio.RescaleIntensity(out_min_max=(-1, 1)), 
    tio.CropOrPad(target_shape=(72, 72, 24))
])

TRAIN_TRANSFORMS = tio.Compose([
    tio.RandomFlip(axes=(1), flip_probability=0.5),
])

class PFOAInDataset_T1W(Dataset):
    def __init__(self, root_dir: str):
        super().__init__()
        self.image_dir = os.path.join(root_dir, 'images')
        self.label_dir = os.path.join(root_dir, 'labels')
        self.preprocessing = PREPROCESSING_TRANSORMS
        self.image_paths = self.get_image_files()


    def get_image_files(self):
        nifti_file_names = os.listdir(self.image_dir)
        folder_names = [os.path.join(
            self.image_dir, nifti_file_name) for nifti_file_name in nifti_file_names if
            nifti_file_name.endswith('.nii.gz')]
        
        test = ['1-2165987L.nii.gz', '1-2235585L.nii.gz', '1-2306902L.nii.gz', '1-2456542L.nii.gz', 
                '1-2568128L.nii.gz', '1-2674487L.nii.gz', '1-2776363L.nii.gz', '1-2973561L.nii.gz', 
                '1-3030469L.nii.gz', '1-3082421L.nii.gz', '1-3151401L.nii.gz', '1-3277730L.nii.gz', 
                '1-3381425L.nii.gz', '1-3554526L.nii.gz', '1-3593833L.nii.gz', '1-3627077L.nii.gz', 
                '1-3691706R.nii.gz', '1-3772881L.nii.gz']
        
        folder_names_filtered = [folder_name for folder_name in folder_names if
                                 os.path.basename(folder_name) in test]
        return folder_names_filtered

    def __len__(self):
        return len(self.image_paths)

    def train_transform(self, image, label, p):
        TRAIN_TRANSFORMS = tio.Compose([
            tio.RandomFlip(axes=(1), flip_probability=p),
        ])
        image = TRAIN_TRANSFORMS(image)
        label = TRAIN_TRANSFORMS(label)
        return image, label

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        img = tio.ScalarImage(image_path)

        image_name = image_path.split('/')[-1]
        mask_path = os.path.join(self.label_dir, image_name)
        mask = tio.LabelMap(mask_path)  

        img = self.preprocessing(img)
        mask = self.preprocessing(mask)

        return {
            'GT': img.data.permute(0, -1, 1, 2).repeat(2, 1, 1, 1),
            'GT_name': image_name,
            'gt_keep_mask': mask.data.permute(0, -1, 1, 2),
            'affine': img.affine
        }



