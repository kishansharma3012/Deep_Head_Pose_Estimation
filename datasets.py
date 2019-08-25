import os
import numpy as np
import cv2
import pandas as pd

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from tqdm import tqdm

from PIL import Image, ImageFilter
import utils
from joblib import Parallel, delayed
np.random.seed(0)


def get_list_from_filenames(file_path):
    # input:    relative path to .npy file with file names
    # output:   list of relative path names
    lines = np.load(file_path)
    return lines

class Pose_300W_LP(Dataset):
    # Head pose from 300W-LP dataset
    def __init__(self, data_dir, num_bins, filename_path, transform, debug = 'False', img_ext='.jpg', annot_ext='.mat', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        filename_list = get_list_from_filenames(filename_path)
        
        if debug == 'True':
            filename_list = filename_list[:80]
        self.num_bins = num_bins
        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.X_train[index].split('.')[0] + self.img_ext))
        img = img.convert(self.image_mode)
        mat_path = os.path.join(self.data_dir, self.y_train[index].split('.')[0] + self.annot_ext)

        # Crop the face loosely
        pt2d = utils.get_pt2d_from_mat(mat_path)
        x_min = min(pt2d[0,:])
        y_min = min(pt2d[1,:])
        x_max = max(pt2d[0,:])
        y_max = max(pt2d[1,:])

        # k = 0.2 to 0.40
        k = np.random.random_sample() * 0.2 + 0.2
        x_min -= 0.6 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 0.6 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # We get the pose in radians
        pose = utils.get_ypr_from_mat(mat_path)
        # And convert to degrees.
        pitch = pose[0] * 180 / np.pi
        yaw = pose[1] * 180 / np.pi
        roll = pose[2] * 180 / np.pi

        # Flip?
        rnd = np.random.random_sample()
        if rnd < 0.5:
            yaw = -yaw
            roll = -roll
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Blur?
        rnd = np.random.random_sample()
        if rnd < 0.05:
            img = img.filter(ImageFilter.BLUR)

        # Bin values:  200 bins 
        # 200 bins - Bin width 1 ; 40 bins - bin width 5; 66 bins - Bin Width 3 
        bin_width = int((102 - (-99))/self.num_bins)
        bins = np.array(range(-99, 102, bin_width))
        binned_pose = np.digitize([yaw, pitch, roll], bins) - 1

        # Get target tensors
        labels = binned_pose
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            img = self.transform(img)

        return img, labels, cont_labels, self.X_train[index]

    def __len__(self):
        # 122,450
        return self.length

class AFLW2000(Dataset):
    def __init__(self, data_dir, num_bins, filename_path, transform, img_ext='.jpg', annot_ext='.mat', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext
        self.num_bins = num_bins

        filename_list = get_list_from_filenames(filename_path)
        
        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.X_train[index].split('.')[0] + self.img_ext))
        img = img.convert(self.image_mode)
        mat_path = os.path.join(self.data_dir, self.y_train[index].split('.')[0] + self.annot_ext)
        # Crop the face loosely
        pt2d = utils.get_pt2d_from_mat(mat_path)

        x_min = min(pt2d[0,:])
        y_min = min(pt2d[1,:])
        x_max = max(pt2d[0,:])
        y_max = max(pt2d[1,:])

        k = 0.20
        x_min -= 2 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 2 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # We get the pose in radians
        pose = utils.get_ypr_from_mat(mat_path)
        # And convert to degrees.
        pitch = pose[0] * 180 / np.pi
        yaw = pose[1] * 180 / np.pi
        roll = pose[2] * 180 / np.pi
        
        bin_width = int((102 - (-99))/self.num_bins)
        bins = np.array(range(-99, 102, bin_width))
        bins = np.array(range(-99, 102, bin_width))
        
        labels = torch.LongTensor(np.digitize([yaw, pitch, roll], bins) - 1)
            
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            img = self.transform(img)

        return img, labels, cont_labels, self.X_train[index]

    def __len__(self):
        # 2,000
        return self.length

def Image_File_Extract(folder):
    ext = tuple(['.jpg', '.JPG', '.JPEG', '.png'])
    files = os.listdir(folder)
    file_path_list = []
    count = 0 
    for i in tqdm(range(len(files))):
        file_path = files[i]
        if file_path.endswith(ext):
            mat_file = file_path.split('.')[0] + '.mat'
            mat_path = os.path.join(folder, mat_file)
            # We get the pose in radians
            pose = utils.get_ypr_from_mat(mat_path)
            # And convert to degrees.
            pitch = pose[0] * 180 / np.pi
            yaw = pose[1] * 180 / np.pi
            roll = pose[2] * 180 / np.pi

            
            if  pitch < -99 or pitch > 99 or yaw < -99 or yaw > 99 or roll < -99 or roll > 99:
                count += 1
                continue
            else:
                file_path_list.append(file_path)
    print('Files Removed : ', count)
    print('Files in list: ', len(file_path_list))
    return file_path_list

def save_file_path(path_list, filename):
    filename_list = os.path.join(os.getcwd(),filename)
    path_list = [d.split('.')[0] for d in path_list]
    np.save(filename_list, path_list)
    
if __name__ == '__main__':
    
    ## Preprocessing dataset, extracting and filtering out filenames
    train_dir = os.path.join(os.getcwd(), "datasets", "300W_LP")
    Folder_names = ['HELEN', 'AFW', 'LFPW', 'IBUG', 'HELEN_Flip', 'AFW_Flip', 'LFPW_Flip', 'IBUG_Flip']

    filepath_list_300W_LP = []
    def process_folder(i):
        folder = Folder_names[i]
        folder_path = os.path.join(train_dir, folder)
        print('Processing folder .....', folder_path)
        file_list = Image_File_Extract(folder_path)
        fol_name = folder + '/{0}'
        file_list = [fol_name.format(file_) for file_ in file_list]
        return file_list 
    
    # Parallelize extraction of file names from different folders
    filepath_list_300W_LP.append(Parallel(n_jobs=8)(delayed(process_folder)(i) for i in range(len(Folder_names))))
    filepath_list_300W_LP = [item for sublist in filepath_list_300W_LP[0] for item in sublist]

    train_val_dev_split = [0.6, 0.2, 0.2]
    np.random.shuffle(filepath_list_300W_LP)
    train_file_path = filepath_list_300W_LP[:int(train_val_dev_split[0]*len(filepath_list_300W_LP))]
    val_file_path = filepath_list_300W_LP[int(train_val_dev_split[0]*len(filepath_list_300W_LP)): int((train_val_dev_split[0] + train_val_dev_split[1])*len(filepath_list_300W_LP))]
    dev_file_path = filepath_list_300W_LP[int((train_val_dev_split[0] + train_val_dev_split[1]) *len(filepath_list_300W_LP)):]
     
    save_file_path(train_file_path, 'train_filename_all.npy')
    save_file_path(val_file_path, 'val_filename_all.npy')
    save_file_path(dev_file_path, 'dev_filename_all.npy')
    print('Total images :', len(filepath_list_300W_LP))
    print('Train images :', len(train_file_path))
    print('Val images :', len(val_file_path))
    print('dev images :', len(dev_file_path))
    
    # Test Filename list 
    test_dir = os.path.join(os.getcwd(), "datasets", "AFLW2000-3D","AFLW2000")
    
    test_file_path = Image_File_Extract(test_dir)
    save_file_path(test_file_path, 'test_filename.npy')
    print('Test images :', len(test_file_path))
    
    print('Processing Done !!!')
    