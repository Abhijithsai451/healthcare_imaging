import os
import shutil
from glob import glob

import dicom2nifti
import dicom2nifti.settings as settings
import nibabel as nib
import numpy as np
from monai.data import Dataset, DataLoader, CacheDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    ToTensord,
    AddChanneld,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    Resized, Orientationd
)
from monai.utils import set_determinism, first
import matplotlib.pyplot as plt
import myconfig

settings.disable_validate_slice_increment()
# Properties
input_path = myconfig.input_path
output_path = myconfig.output_path
number_slices = myconfig.number_slices
sliced_images = myconfig.sliced_images
sliced_labels = myconfig.sliced_labels
nifti_images = myconfig.nifti_images
nifti_labels = myconfig.nifti_labels


# Creating groups of 64 Slices
def create_groups(input_path, output_path, number_slices):
    for patient in glob(input_path + '/*'):
        patient_name = os.path.basename(os.path.normpath(patient))
        number_folders = int(len(glob(patient + '/*')) / number_slices)

        for i in range(number_folders):
            output_path_name = os.path.join(output_path, patient_name + '_' + str(i))
            os.mkdir(output_path_name)
            for i, file in enumerate(glob(patient + '/*')):
                if i == number_slices:
                    break
                shutil.move(file, output_path_name)


# Convert the Dicom files into Nifties
def dicom_to_nifti(sliced_images, sliced_labels):
    list_images = glob(sliced_images + '/*')
    list_labels = glob(sliced_labels + '/*')
    for patient in list_images:
        patient_name = os.path.basename(os.path.normpath(patient))
        dicom2nifti.dicom_series_to_nifti(patient, os.path.join(nifti_images, patient_name + '.nii'))

    for patient in list_labels:
        patient_name = os.path.basename(os.path.normpath(patient))
        dicom2nifti.dicom_series_to_nifti(patient, os.path.join(nifti_labels, patient_name + '.nii'))


def delete_empty_slices(in_dir):
    # Load every nifti files and check the slices with empty data.
    # Remove the remove slices.
    # Using nibabel to read the images
    list_patients = []
    for patient in glob(os.path.normpath(in_dir + '/*')):
        img = nib.load(patient)

        if len(np.unique(img.get_fdata())) > 2:
            print(os.path.basename(os.path.normpath(patient)))
            list_patients.append(os.path.basename(os.path.normpath(patient)))
    print(list_patients)
    return list_patients
#list_patients = delete_empty_slices(myconfig.nifti_images)


def prepare_data(in_dir, pixdim=(1.5, 1.5, 1.0), a_min=-200, a_max=200, spatial_size=[128, 128, 128], cache=False):
    # This is a function to preprocess the data. We take the data after removing the empty slices and apply
    # preprocessing functions like transformations
    set_determinism(seed=0)

    train_images_path = sorted(glob(os.path.join(in_dir, 'images', '*.nii')))
    train_labels_path = sorted(glob(os.path.join(in_dir, 'labels', '*.nii')))

    test_images_path = sorted(glob(os.path.join(in_dir, 'images', '*.nii')))
    test_labels_path = sorted(glob(os.path.join(in_dir, 'labels', '*.nii')))

    train_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in
                   zip(train_images_path, train_labels_path)]
    test_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in
                  zip(test_images_path, test_labels_path)]

    transformations = Compose(
        [
            LoadImaged(keys=["vol", "seg"]),
            AddChanneld(keys=["vol", "seg"]),
            Spacingd(keys=["vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
            Orientationd(keys=["vol", "seg"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=["vol", "seg"], source_key="vol"),
            Resized(keys=["vol", "seg"], spatial_size=spatial_size),
            ToTensord(keys=["vol", "seg"]),
        ]
    )
    """test_transforms = Compose(
        [
            LoadImaged(keys=["vol", "seg"]),
            AddChanneld(keys=["vol", "seg"]),
            Spacingd(keys=["vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
            Orientationd(keys=["vol", "seg"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=['vol', 'seg'], source_key='vol'),
            Resized(keys=["vol", "seg"], spatial_size=spatial_size),
            ToTensord(keys=["vol", "seg"]),

        ]
    )"""

    if cache:
        train_ds = CacheDataset(data=train_files, transform=transformations)
        train_loader = DataLoader(train_ds, batch_size=1)

        test_ds = CacheDataset(data=test_files, transform=transformations)
        test_loader = DataLoader(test_ds, batch_size=1)

        return train_loader, test_loader
    else:
        train_ds = Dataset(data=train_files, transform=transformations)
        train_loader = DataLoader(train_ds, batch_size=1)

        test_ds = Dataset(data=test_files, transform=transformations)
        test_loader = DataLoader(test_ds, batch_size=1)

        return train_loader, test_loader

def viz_sample_data(data, SLICE_NUMBER=1, train = True, test = False):

    check_train_data, check_test_data = data

    view_train_data = first(check_train_data)
    view_test_data = first(check_test_data)

    if train:
        plt.figure("visualization Train", (12,6))
        plt.subplot(1,2,1)
        plt.title(f"vol {SLICE_NUMBER}")
        plt.imshow(view_train_data["vol"][0,0, :,:, SLICE_NUMBER],cmap='gray')

        plt.subplot(1,2,2)
        plt.title(f"seg {SLICE_NUMBER}")
        plt.imshow(view_train_data["seg"][0,0, :,:, SLICE_NUMBER])
        plt.show()
    if test:
        plt.figure("visualization Test", (12, 6))
        plt.subplot(1, 2, 1)
        plt.title(f"vol {SLICE_NUMBER}")
        plt.imshow(view_test_data["vol"][0, 0, :, :, SLICE_NUMBER], cmap='gray')

        plt.subplot(1, 2, 2)
        plt.title(f"seg {SLICE_NUMBER}")
        plt.imshow(view_test_data["seg"][0, 0, :, :, SLICE_NUMBER])
        plt.show()


sample = prepare_data(myconfig.nifti_base)
viz_sample_data(sample,85)

