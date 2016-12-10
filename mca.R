require(FactoMineR)

train <- read.csv('~/Downloads/train_split/train_a_no_dup.csv')
sig_cat = read.csv('~/Downloads/sig_cat_col_bonf.csv')
sig_cat <- sig_cat[1:31]
sigcat <- colnames(sig_cat)
train <- train[,sigcat]
train[sigcat] <- lapply(train[sigcat], factor)
mca <- MCA(train)