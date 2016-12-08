library(dplyr)
library(ggplot2)
library(data.table)
library(PCAmixdata)

scat <- fread('../train_a_cleaned.csv.cat')
snumer <- fread('../train_a_cleaned.csv.numer')

scat <- lapply(scat, as.factor)
scat <- as.data.frame(scat)

pca <- PCAmix(snumer, scat, rename.level=TRUE)
