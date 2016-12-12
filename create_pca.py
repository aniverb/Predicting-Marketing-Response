import numpy as np
from sklearn.decomposition import PCA
import sys
import pandas as pd
from sklearn.externals import joblib
import matplotlib
import matplotlib.pyplot as plt

'''
python create_pca <csv>
'''

N_KEEP = 200

df = pd.read_csv(sys.argv[1], header=0, index_col=False)
df_norm = (df - df.mean()) / (df.max() - df.min())
pca = PCA(svd_solver='full', n_components=N_KEEP)
df_norm = pca.fit_transform(df_norm)
pd.DataFrame(data=df_norm, columns=['PC' + str(i) for i in range(N_KEEP)]).to_csv(sys.argv[1] + '.projected', index=False)
joblib.dump(pca, sys.argv[1] + '.pcapk' )

# Remove n_components before this
#plt.plot(pca.explained_variance_ratio_[:30])
#plt.xlabel('Principal Component')
#plt.ylabel('Ratio of Explained Variance')
#plt.title('PCA of Numerical Data')
#plt.savefig(sys.argv[1] + '_pca.png', bbox_inches='tight')