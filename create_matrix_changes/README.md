Scripts to generate the matrix of changes from the phylogenetic tree and an MSA

The R version is: 4.0.5
The libraries needed are:
- phytools
- phangorn

## Installation
```r
install.packages("remotes")
install_version("phytools", version = "0.7.70", repos = "http://cran.us.r-project.org")
install_version("phangorn", version = "2.6.3", repos = "http://cran.us.r-project.org")
```


The Python version is: 3.6.8
The libraries needed are:
- numpy==1.19.5
- pandas==1.1.5

bash run_ancestral_changes_matrix.sh my_fasta.fasta my_tree.nwk output_folder amino
