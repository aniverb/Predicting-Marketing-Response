import numpy as np
from sklearn.decomposition import PCA
import sys
import pandas as pd
from sklearn.externals import joblib
import matplotlib
import matplotlib.pyplot as plt


df = pd.read_csv(sys.argv[1], header=0)
df_norm = (df - df.mean()) / (df.max() - df.min())
pca = PCA(svd_solver='full')
pca.fit(df_norm)
#joblib.dump(pca, sys.argv[1] + '.pca')

# To load mdel
#pca = joblib.load(sys.argv[1] + '.pca')
plt.plot(pca.explained_variance_ratio_[:30])
plt.xlabel('Principal Component')
plt.ylabel('Ratio of Explained Variance')
plt.title('PCA of Numerical Data')
plt.savefig(sys.argv[1] + '_pca.png', bbox_inches='tight')
plt.show()