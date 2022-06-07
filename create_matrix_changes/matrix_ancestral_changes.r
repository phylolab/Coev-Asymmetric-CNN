#library(ape)
library(phytools)
library(phangorn)

get_change <- function(orig_max, ancestral_max, list_msa) {
  name_change <- paste(toString(orig_max), toString(ancestral_max))
  return(list_msa[name_change][[1]])
}

#It is saving the type of change from one amino acid or, nucleotide, to the other,
get_mapping <- function(type_msa){
  aux_vec <- list()
  count_aux <- 1
  for(i in 1:type_msa){
    for(j in 1:type_msa){
      name_var <- paste(toString(i), toString(j))
      if( i == j){
        aux_vec[name_var] <- 0
      }
      else{
        aux_vec[name_var] <- count_aux
        count_aux <- count_aux + 1
      }
    }
  }
  return(aux_vec)
}

get_intermediate_pos_matrix <- function(orig_matrix, ancest_matrix) {
  matrix_len <- length(orig_matrix[,1]);
  aux_array <- rep(0,matrix_len);

  for(row_i in 1:length(orig_matrix[,1])){
    max_orig <- which.max(orig_matrix[row_i,]);
    max_ancest <- which.max(ancest_matrix[row_i,]);
    if (max_orig != max_ancest){
      aux_array[row_i] <- 1;#max_orig;
    }
  }
  return(aux_array)
}

# with the position
get_intermediate_values_matrix <- function(orig_matrix, ancest_matrix, mapping_change) {
  matrix_len <- length(orig_matrix[,1]);
  aux_array <- rep(0,matrix_len);
  for(row_i in 1:length(orig_matrix[,1])){
    max_orig <- which(orig_matrix[row_i,] == max(orig_matrix[row_i,]))
    #print("orig")
    #print(max_orig)
    max_ancest <- which(ancest_matrix[row_i,] == max(ancest_matrix[row_i,]))
    #print("ancest")
    #print(max_ancest)
    if (is.na(match(max_ancest, max_orig))){
      #print(" NOT Equal")
      change_msa <- get_change(max_orig, max_ancest, mapping_change)
    }
    else{
      change_msa <- 0
    }
    aux_array[row_i] <- change_msa;
  }
  return(aux_array)
}
