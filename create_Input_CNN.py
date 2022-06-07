import numpy as np
import pandas as pd
import os
import h5py


def dict_counter(a):
    """
    Parameters: 
    a (numpy.ndarray): a list of numbers
    
    Returns:
    dict_counter (dictionary): where the keys are the numbers in a, 
                              and the values their frequency
    """

    dict_counter = {}
    for i in a:
        if i not in dict_counter:
            counter = np.where(a == i)
            dict_counter[i] = len(counter[0])

    return dict_counter

def convert_to_dim_input_weigth(input_data, row_num, col_num):
    """ Convert a matrix of shape AxB in a matrix AxBx381
        the matrix AxB can have numbers in the range [0-381]
        this function generate a matrix of 381 layers, where each layer correspond to the number
        in the matrix AxB and it will have the frequency of that number in that row.
        For example:
            if we have the next matrix (4x4):
                1 4 2 2
                0 0 0 1
                1 1 1 0
                0 0 1 2
            we will generate a matrix (4x4x381):
                we will have 381 layers, and the layer 1 would be:
                    1 0 0 0
                    0 0 0 1
                    3 3 3 0
                    0 0 1 0
                layer 2 would be:
                    0 0 2 2
                    0 0 0 0
                    0 0 0 0
                    0 0 0 1
                layer 3 would be full of zeros
                layer 4 would be:
                    0 1 0 0
                    0 0 0 0
                    0 0 0 0
                    0 0 0 0
                and the rest of the layers would be full of zeros too.
                
    Parameters:
    input_data (numpy.ndarray): a matrix of size AxB
    
    row_num (int): number of rows
    
    col_num (int): number of columns
    
    Returns:
    aux_matrix (numpy.ndarray): matrix of size AxBx381
    
    """
    # first we create a matrix full of zeros 
    # and we add a new dimension to be able to add the other 381 dimensions
    aux_matrix = np.zeros((row_num,col_num))
    aux_matrix = np.expand_dims(aux_matrix, axis=2)

    full_matrix = np.zeros((row_num, col_num, 381), dtype='int16')
    #if count == 0:
    #    print("Size of full_matrix: %d %d %d" %(row_num, col_num, 381))

    for j in range(1,381): # Checking each type of change

        # we look for the number j in the input matrix
        # places will have the rows where j is in the input matrix
        places = np.where(input_data == j)

        # if the number is in the matrix then:
        if (len(places[0]) > 0):

            #print("  column %3d has %3d elements" %(j, len(places[0])))

            # create a dictionary to contain the frequency of these numbers
            dict_aux = dict_counter(places[0])

            # create a list of list: 
            #  each list inside the main list will have 2 values
            #  the coordinates of the row and column where the number j is

            c = list(map(lambda x, y: [x,y], places[0], places[1]))

            # for each pair of values row_column, we will fill in our aux_layer
            # i[0] is the row
            # i[1] is the column
            # dict_aux[i[0]] is the frequency of j in that row i[0]

            for i in c:
                full_matrix[i[0], i[1], j-1] = dict_aux[i[0]]
                
    np.set_printoptions(threshold=np.inf)
    
    return full_matrix

def create_dataset(row_size, col_size, folder_input, folder_output_name = '.', hdf5_format = True):
    ###
    # SETTINGS
    ###

    #folder_input = '/scratch/rramabal/CNN_project_weeks/SelectomeCNN/chapter_3_DATA/DataSet_Freq/cnnData'


    ## Train dataset
    training_dataset = []
    training_file_name = []

    ## Testing dataset
    diff_dataset = []
    diff_filename = []

    row_m = row_size # 199
    col_m = col_size # 396

    ## For training
    num_train_coev = 0
    print("Training dataset")
    for i in os.listdir(folder_input+'/train/COEV/'):

        if ('DS' not in i) & ('MAP' not in i):
            #print(i)
            try:
                df = pd.read_csv(folder_input+'/train/COEV/'+i, index_col=0, header=None, skiprows=1)
                matrix = df.to_numpy()
                if matrix.shape[0] <= row_m:
                    id_file = i #.split('.')[0] #int(i.split('_')[2])
                    training_file_name.append(id_file)

                    new_m = convert_to_dim_input_weigth(matrix, row_m, col_m)
                    #if count == 0:
                    #    print("Size of new_m = %d Bytes" %(sys.getsizeof(new_m)))

                    one_layer_m = np.sum(new_m, axis=2)
                    training_dataset.append(one_layer_m)

                    #training_dataset.append(new_m) #.tolist()
                    num_train_coev += 1
                    break
            except:
                print(i)

    print("Coev DONE")

    ### NO COEVOLUTION
    num_train_non_coev = 0
    for i in os.listdir(folder_input+'/train/NO_COEV/'):

        if ('DS' not in i) & ('MAP' not in i):
            df = pd.read_csv(folder_input+'/train/NO_COEV/'+i, index_col=0, header=None, skiprows=1)

            matrix = df.to_numpy()
            if matrix.shape[0] <= row_m:
                id_file = i #.split('.')[0] #int(i.split('_')[2])
                training_file_name.append(id_file)

                new_m = convert_to_dim_input_weigth(matrix, row_m, col_m)

                one_layer_m = np.sum(new_m, axis=2)
                training_dataset.append(one_layer_m)

                #training_dataset.append(new_m) #.tolist()
                num_train_non_coev += 1
                break

    print("No Coev DONE")

    ######
    #####
    ## ONES are COEVOLUTION
    ## ZEROS are NOT COEVOLUTION
    #####
    ######

    label_train_coev = np.ones(num_train_coev)
    label_train_non_coev = np.zeros(num_train_non_coev, dtype=int)
    label_train = np.concatenate((label_train_coev, label_train_non_coev), axis=None)

    training_dataset = np.float64(training_dataset)
    #print(training_dataset.shape)


    ## Test dataset
    print("Testing dataset")
    diff_train_coev = 0
    for i in os.listdir(folder_input+'/test/COEV/'):
        if ('DS' not in i) & ('MAP' not in i):
            try:
                df = pd.read_csv(folder_input+'/test/COEV/'+i, index_col=0, header=None, skiprows=1)
                matrix = df.to_numpy()

                if matrix.shape[0] <= row_m:
                    id_file = i #.split('.')[0] #int(i.split('_')[2])
                    diff_filename.append(id_file)

                    new_m = convert_to_dim_input_weigth(matrix, row_m, col_m)

                    one_layer_m = np.sum(new_m, axis=2)
                    diff_dataset.append(one_layer_m)

                    #diff_dataset.append(new_m.tolist())
                    diff_train_coev += 1
                    break
            except:
                print(i)

    print("Coev DONE")


    ## Testing dataset under non coevolution
    diff_train_non_coev = 0 
    for i in os.listdir(folder_input+'/test/NO_COEV/'):
        if ('DS' not in i) & ('MAP' not in i):
            df = pd.read_csv(folder_input+'/test/NO_COEV/'+i, index_col=0, header=None, skiprows=1)
            matrix = df.to_numpy()
            if matrix.shape[0] <= row_m:
                id_file = i #.split('.')[0] #int(i.split('_')[2])
                diff_filename.append(id_file)

                new_m = convert_to_dim_input_weigth(matrix, row_m, col_m)

                one_layer_m = np.sum(new_m, axis=2)
                diff_dataset.append(one_layer_m)

                #diff_dataset.append(new_m.tolist())
                diff_train_non_coev += 1
                break

    print("No Coev DONE")

    diff_dataset = np.float64(diff_dataset)

    ######
    #####
    ## ONES are COEVOLUTION
    ## ZEROS are NOT COEVOLUTION
    #####
    ######

    diff_label_train_coev = np.ones(diff_train_coev)
    diff_label_train_non_coev = np.zeros(diff_train_non_coev, dtype=int)
    diff_label_train = np.concatenate((diff_label_train_coev, diff_label_train_non_coev), axis=None)

    if hdf5_format:
        #folder_output = '/scratch/rramabal/CNN_project_weeks/SelectomeCNN/chapter_3_DATA/'

        #hf_all = h5py.File(folder_output + 'Freq_simulated_Selectome_199_396_BIG.h5', 'w')
        hf_all = h5py.File(folder_output_name+'output.h5', 'w')

        hf_all.create_dataset('training_dataset', data=training_dataset)
        hf_all.create_dataset('label_train', data=label_train)
        hf_all.create_dataset('file_names_train', data=training_file_name)

        hf_all.create_dataset('testing_dataset', data=diff_dataset)
        hf_all.create_dataset('label_test', data=diff_label_train)
        hf_all.create_dataset('file_names_test', data=diff_filename)
        hf_all.close()

    else:
        asdf = np.array(training_dataset)
        asdf_name = np.array(training_file_name)

        np.save(folder_output_name+'training_file.npy', asdf)
        np.save(folder_output_name+'name_training_file.npy', asdf_name)	
        np.save(folder_output_name+'label_file.npy', label_train)


        asdf = np.array(diff_dataset)
        asdf_name = np.array(diff_filename)

        np.save(folder_output_name+'testing_file.npy', asdf)
        np.save(folder_output_name+'name_testing_file.npy', asdf_name)
        np.save(folder_output_name+'label_test_file.npy', diff_label_train)


