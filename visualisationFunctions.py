from scipy.spatial import distance
from scipy.stats.stats import pearsonr
from scipy import stats   
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

def read_coev_sites(my_path, my_name):

    aux_df = pd.read_csv(my_path + my_name)
    pos_1 = int(aux_df[aux_df['New'] == 0]['Original'])
    pos_2 = int(aux_df[aux_df['New'] == 1]['Original'])
    #print(pos_1, pos_2)

    pos_3 = int(aux_df[aux_df['New'] == 2]['Original'])
    pos_4 = int(aux_df[aux_df['New'] == 3]['Original'])
    #print(pos_3, pos_4)
    
    return((pos_1,pos_2), (pos_3, pos_4))

"""
Hamming: closer to 0 the best
Pearson corr: closer to 1 the best
Jaccard: closer to 0 the best
Spearman: closer to 1 the best
"""
def calculate_distance(my_matrix, site_1, site_2, dist='pearson'):
    if dist == 'pearson':
        return pearsonr(my_matrix[site_1], my_matrix[site_2])
    elif dist == 'hamming':
        return distance.hamming(my_matrix[site_1], my_matrix[site_2])
    elif dist == 'spearman':
        return stats.spearmanr(my_matrix[site_1], my_matrix[site_2], axis=None)
    else:
        return distance.jaccard(my_matrix[site_1], my_matrix[site_2])

"""
Check common branches
"""
def check_common(my_sites, my_x):
    max_one = my_x[my_sites[0][0]]
    max_two = my_x[my_sites[1][0]]
    indices_one = [index for index, element in enumerate(max_one) if element == int(max(max_one))]
    #print(indices_x)
    indices_two = [index for index, element in enumerate(max_two) if element == int(max(max_two))]
    #print(indices_y)
    
    return list(set(indices_one).intersection(indices_two))

def check_common_2(my_sites, my_x, pairs):
    out_max = []
    my_max = -100
    for i in range(0, int(pairs/2)+1, 2):
        max_one = my_x[my_sites[i][0]]
        max_two = my_x[my_sites[i+1][0]]
        indices_one = [index for index, element in enumerate(max_one) if element == int(max(max_one))]
        #print(indices_x)
        indices_two = [index for index, element in enumerate(max_two) if element == int(max(max_two))]
        #print(indices_y)
        output = list(set(indices_one).intersection(indices_two))
        if len(output) > my_max:
            my_max = len(output)
            out_max = output
    
    return out_max

# Get pair of sites
def view_pair_sites(model, my_x, number_sites, plot_orig = True):
    x = my_x # x_batch[0] #torch.FloatTensor(x_batch)
    if plot_orig:
        print("Original matrix")
        plt.figure(figsize = (60,100))
        plt.imshow(x)
        plt.show()

    x_aux = np.expand_dims(x, axis=0)
    x_aux = torch.FloatTensor(x_aux)
    pred_x_conv1 = F.relu(model.conv1(x_aux))

    data_x = []
    for i in range(len(pred_x_conv1[0])):
        data_x.append(list(pred_x_conv1[0][i][0].detach().numpy()))

    max_1 = -100
    my_filter = ''
    for my_filter_1 in data_x:
        if sum(my_filter_1) > max_1:
            max_1 = sum(my_filter_1)
            my_filter = my_filter_1

    num_col = len(my_filter)
    #print(num_col)
    arr_my_filter = np.array(my_filter)

    ind = (-arr_my_filter).argsort()[:10]
    top4 = arr_my_filter[ind]

    sol_aux = x_aux
    for aux_filter in data_x:
        asd = x_aux*np.array([aux_filter])
        sol_aux = np.concatenate((sol_aux, asd), axis=1)

    #print(sol_aux.shape)
    sol_aux = np.delete(sol_aux, 0, axis=1)
    #print(sol_aux.shape)


    #x_aux = np.expand_dims(x_batch, axis=0)
    x_asdf = torch.FloatTensor(sol_aux)
    #print(x_asdf.shape)
    pred_x_conv2 = F.relu(model.conv2(x_asdf))
    #print(pred_x_conv2.shape)
    #print(len(pred_x_conv2[0]))

    data_y = []
    for i in range(len(pred_x_conv2[0])):
        #print(len(pred_x_conv2[0][i]))
        my_out = pred_x_conv2[0][i].detach().numpy()
        asdf = [float(i) for i in my_out]
        data_y.append(asdf)

    data_y = np.array(data_y)

    max_2 = -100
    my_filter_sites = ''
    for my_filter_1 in data_y:
        if sum(my_filter_1) > max_2:
            max_2 = sum(my_filter_1)
            my_filter_sites = my_filter_1
    #print(my_filter)

    my_filter_sites_ENUM = enumerate(my_filter_sites)
    list2 = sorted(my_filter_sites_ENUM, key=lambda z:z[1])
    #print(list2[::-1][0:number_sites])

    final_list = list2[::-1][0:number_sites]
    final_list = [j[0] for j in final_list]
    my_numpy_rows = []

    for aux_row in range(len(x)):
        if aux_row in final_list:
            my_numpy_rows.append(x[aux_row].detach().numpy())

    #print(np.array(my_numpy_rows))

    #plt.figure(figsize = (60,100))
    #plt.imshow(np.array(my_numpy_rows), interpolation='nearest')
    #plt.show()
    return list2[::-1][0:number_sites]

    """# Real Data
    my_dim_sites = np.reshape(my_filter_sites, (199, 1))
    my_dim_branch = np.reshape(my_filter, (1, 396))
    #print(my_dim_sites.shape, my_dim_branch.shape)
    new_sol = np.dot(my_dim_sites, my_dim_branch)"""
