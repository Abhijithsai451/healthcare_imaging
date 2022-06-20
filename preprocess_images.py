import os
from glob import glob
import shutil
import myconfig
import dicom2nifti
import dicom2nifti.settings as settings

settings. disable_validate_slice_increment()
# Properties
input_path = myconfig.input_path
output_path = myconfig.output_path
number_slices = myconfig.number_slices
sliced_images = myconfig.sliced_images
sliced_labels = myconfig.sliced_labels
nifti_images = myconfig.nifti_images
nifti_labels = myconfig.nifti_labels

# Creating groups of 64 Slices
def create_groups(input_path,output_path,number_slices):
    for patient in glob(input_path + '/*'):
        patient_name = os.path.basename(os.path.normpath(patient))
        number_folders = int(len(glob(patient+'/*'))/number_slices)

        for i in range(number_folders):
            output_path_name = os.path.join(output_path,patient_name + '_'+str(i))
            os.mkdir(output_path_name)
            for i, file in enumerate(glob(patient +'/*')):
                if i == number_slices:
                    break
                shutil.move(file, output_path_name)

# Convert the Dicom files into Nifties
def dicom_to_nifti():
    list_images = glob(sliced_images + '/*')
    list_labels = glob(sliced_labels + '/*')
    for patient in list_images:
        patient_name = os.path.basename(os.path.normpath(patient))
        dicom2nifti.dicom_series_to_nifti(patient,os.path.join(nifti_images,patient_name+'.nii'))

    for patient in list_labels:
        patient_name = os.path.basename(os.path.normpath(patient))
        dicom2nifti.dicom_series_to_nifti(patient,os.path.join(nifti_labels,patient_name+'.nii'))



