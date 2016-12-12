import numpy as np
from sklearn.decomposition import PCA
import sys
import pandas as pd
from sklearn.externals import joblib
import matplotlib
import matplotlib.pyplot as plt


df = pd.read_csv(sys.argv[1], header=0, index_col=False)
df_norm = (df - df.mean()) / (df.max() - df.min())
pca = PCA(svd_solver='full')
df_norm = pca.fit(df_norm)

SCALE = 500
plt.axis([0,500,0,1.2])
plt.plot(np.cumsum(pca.explained_variance_ratio_[:SCALE]))
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance')
plt.title('PCA of Numerical Data (First ' + str(SCALE) + '/' + str(len(pca.explained_variance_ratio_)) + ')')
plt.savefig(sys.argv[1] + '_pca_zoom.png', bbox_inches='tight')