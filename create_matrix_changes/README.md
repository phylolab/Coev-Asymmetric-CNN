Scripts to generate the matrix of changes from the phylogenetic tree and an MSA

The libraries needed for R:
install_version("phytools", version = "0.7.70", repos = "http://cran.us.r-project.org")

## Installation
```r
install.packages("remotes")
install_version("phytools", version = "0.7.70", repos = "http://cran.us.r-project.org")
install_version("phangorn", version = "2.6.3", repos = "http://cran.us.r-project.org")
```


bash run_ancestral_changes_matrix.sh my_fasta.fasta my_tree.nwk output_folder amino
