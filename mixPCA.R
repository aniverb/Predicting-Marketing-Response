#this code was not used due to inability to rigorously determine what number of dimensions should be kept
library(compiler)
compilePKGS(TRUE)
setCompilerOptions(suppressAll = TRUE, optimize = 3)
enableJIT(3)
install.packages("/home/rgaddip1/dm/data.table_1.10.0.tar.gz", repos=NULL, type="source")
install.packages("/home/rgaddip1/dm/PCAmixdata_2.2.tar.gz", repos=NULL, type="source")
library(data.table)
library(PCAmixdata)

scat <- fread('../train_a_cleaned.csv.cat')
snumer <- fread('../train_a_cleaned.csv.numer')

scat <- lapply(scat, as.factor)
scat <- as.data.frame(scat)

pca <- PCAmix(snumer, scat, rename.level=TRUE)

save(pca, file='pca.results')
