import numpy as np
import torch
import h5py
from pathlib import Path
from torch.utils import data

def load_data_numpy(file_x, file_y, file_label, file_tx, file_ty, file_tlabel, my_seed = 123):
    
    X = np.load(file_x)
    X = X.astype(np.long)
    Y = np.load(file_y)
    L = np.load(file_label)

    tX = np.load(file_tx)
    tX = tX.astype(np.long)
    tY = np.load(file_ty)
    tL = np.load(file_tlabel)

    X = np.concatenate((X, tX), axis=0)
    Y = np.concatenate((Y, tY), axis=0)
    L = np.concatenate((L, tL), axis=0)

    np.random.seed(my_seed)

    rnd_indx = np.random.choice(range(Y.shape[0]), size=Y.shape[0])

    Yt = Y[rnd_indx] #.reshape((1, Y.shape[0]))
    Xt = X[rnd_indx]
    Lt = L[rnd_indx]

    return Xt, Yt, Lt

class HDF5Dataset(torch.utils.data.Dataset):
    """Represents an abstract HDF5 dataset.
    
    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
    """
    def __init__(self, file_path, train=True):
        super(HDF5Dataset, self).__init__()
        
        if train:
            self.h5_file = h5py.File(file_path , 'r')
            self.dataset = self.h5_file['training_dataset']
            self.label = self.h5_file['label_train']
            self.file_names = self.h5_file['file_names_train']
        else:
            self.h5_file = h5py.File(file_path , 'r')
            self.dataset = self.h5_file['testing_dataset']
            self.label = self.h5_file['label_test']
            self.file_names = self.h5_file['file_names_test']

    def __getitem__(self, index):
        
        # get data
        #print(self.training_dataset[index].shape)
        x = torch.FloatTensor(np.array(self.dataset[index]))

        # get label
        #print("inside function", self.label_train[index])
        y = self.label[index]

        # get name file 
        #print("inside function", self.file_names[index])
        z = self.file_names[index]
        return (x, y, z)
        
            

    def __len__(self):
        return len(self.label)


def hdf_data(hdf5_data_folder, *name_h5, my_seed = 19135279):

    ## Another Try
    #hdf5_data_folder = '/scratch/rramabal/CNN_project_weeks/DATA/'

    #hdf5_data_folder = '/scratch/rramabal/CNN_project_weeks/SelectomeCNN/chapter_3_DATA/'
    #data_hf = [hdf5_data_folder+'no_freq_simulated_Selectome_199_396_Small.h5']
    
    print(name_h5[0][0])
    
    print(len(name_h5))
    print(len(name_h5[0]))
    data_hf = [hdf5_data_folder + name_h5[0][0]]
   
    if len(name_h5[0]) > 1:
        for i in range(1, len(name_h5[0])):
            data_hf.append(hdf5_data_folder + name_h5[0][i])

    first = True
    print(data_hf)
    for index_data_hf in data_hf:

        train_dataset = HDF5Dataset(index_data_hf, train=True)

        np.random.seed(my_seed)
        idx = np.arange(0,len(train_dataset))
        print("SIZE:", len(train_dataset))
        np.random.shuffle(idx)
        #print(idx[0:5])
        #print(len(idx))
        new_len = int(len(idx))
        #print(new_len)
        
        #val_value = int(len(idx)*70/100)
        #test_value = int(len(idx)*20/100)
        
        val_value = int(new_len*90/100)

        if first:
            first = False
            train_data = torch.utils.data.Subset(train_dataset, idx[:val_value])
            val_data = torch.utils.data.Subset(train_dataset, idx[val_value:])
            print('train:', len(train_data), 'val:', len(val_data))
            #print(len(train_data)+len(val_data))
            #print("Train")
            
        else:
            train_data_aux = torch.utils.data.Subset(train_dataset, idx[:val_value])
            val_data_aux = torch.utils.data.Subset(train_dataset, idx[val_value:])
            
            train_data = torch.utils.data.ConcatDataset([train_data, train_data_aux])
            val_data = torch.utils.data.ConcatDataset([val_data, val_data_aux])
            print('train:', len(train_data), 'val:', len(val_data))
            #print(len(train_data)+len(val_data))

        print("")
    
    return train_data, val_data
