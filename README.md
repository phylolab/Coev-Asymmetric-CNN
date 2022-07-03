
# Transorm MSA and phylogenetic tree into a 2D matrix of changes

Folder create_matrix_changes
It contains:
 - the libraries needed 
 - the scripts to run and generate the csv files with the 2D matrixes


# Coev-Asymmetric-CNN

Convolutional Neural Network, with 2 asymmetric 1D convolutions, to detect any signal of coevolution given an input matrix.
 - First convolution is of size Lx1
 - Second convolution is of size 1xB
 
This input matrix may be of size LxB, where L is the number of sites of a given MSA, and B the number of branches of its phylogenetic tree.

