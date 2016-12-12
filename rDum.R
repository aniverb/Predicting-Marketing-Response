dev=read.csv('C:\\Users\\aniverb\\Documents\\Grad_School\\JHU\\436 - Data Mining\\Project\\Springleaf data\\clean\\dev_200.csv')
coln=colnames(dev)[203:228]
catDev=lapply(dev[,coln] , factor)
catDevDum=model.matrix(~.-1, catDev)
write.csv(catDevDum, 'devDum.csv', row.names = F)
