import numpy as np
from sklearn.decomposition import PCA
import sys
import pandas as pd
from sklearn.externals import joblib
import matplotlib
import matplotlib.pyplot as plt


df = pd.read_csv(sys.argv[1], header=0)
pca = PCA(svd_solver='full')
pca.fit_transform(df.T)
df.to_csv('tmp', index=False)
print df.T.shape
joblib.dump(pca, sys.argv[1] + '.pca')



# To load model
pca = joblib.load(sys.argv[1] + '.pca')
plt.plot(pca.explained_variance_[:10])
plt.show()