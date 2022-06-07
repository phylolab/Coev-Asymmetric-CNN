source('matrix_ancestral_changes.r')
  
args <- commandArgs(trailingOnly = TRUE)
name_fasta <- args[1]
name_tree <- args[2]
matrix_path <- args[3]

mapping_list <- get_mapping(20);
length(mapping_list)

print(name_tree)
print(name_fasta)

tree_coev <- read.tree(name_tree)

## Change length 0 to 0.000001
tree_coev$edge.length[which(tree_coev$edge.length==0)] <- 0.000001

#plot(tree_coev)
align <- read.phyDat(name_fasta, format="fasta", type="AA")

fit_coev <- pml(tree_coev, align,optQ=T,optEdge=F,optRate=T)
anc.ml <- ancestral.pml(fit_coev, type = "ml") #, site.pattern=F)
#print(anc.ml)
#plotAnc(tree_coev, anc.ml, i=10, site.pattern=F)

vec <-vector()
count <- 0
for(z in 1:nrow(tree_coev$edge)){

  ancestral_matrix <- anc.ml[[tree_coev$edge[z,1]]][attr(anc.ml,"index"),] #convert_to_matrix(anc.ml[tree_coev$edge[z,1]]);
  orig_matrix <- anc.ml[[tree_coev$edge[z,2]]][attr(anc.ml,"index"),] #convert_to_matrix(anc.ml[tree_coev$edge[z,2]]);
  vec <- cbind(vec, get_intermediate_values_matrix(orig_matrix, ancestral_matrix, mapping_list));
}

print(matrix_path)
write.csv(vec, file = matrix_path)