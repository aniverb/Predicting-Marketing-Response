catcols = read.csv("catcols.csv", header = TRUE)[,-1]
target = read.csv("target.csv", header = FALSE)[,-1]

#Import catcols.csv - Columns + Target
dim(catcols) #72615x770

pvals = matrix(nrow = 1, ncol = ncol(catcols))
#l_five = 0 #Number of features containing a group with freq < 5
#twobytwo = 0 #Number of features with two groups

for (i in 1:ncol(catcols)){
  crosstab = table(catcols[,i], target)
  #if (nrow(crosstab) != 1){ #Some features have only one category
  #  if (nrow(crosstab) == 2){
  #  twobytwo = twobytwo + 1
  #}
  #if (any(crosstab<5)){
  #  l_five = l_five + 1
  #}
  
  pvals[i] = as.numeric(unlist(fisher.test(crosstab, workspace = 2e+8, hybrid = TRUE, simulate.p.value = TRUE)[1]))
}

pvals = rbind(colnames(catcols),pvals)
sig_pvals = pvals[,which(pvals[2,]<0.05)]
num_sig = ncol(sig_pvals)

sprintf("Number of significant features: %d out of %d", num_sig, ncol(catcols))
#sprintf("Number of features containing group with freq < 5: %d out of %d", l_five, ncol(catcols))
#sprintf("Number of features with two groups (2x2 tables): %d out of %d", twobytwo, ncol(catcols))

write.csv(sig_pvals[1,], "catSigColFisher.csv")
