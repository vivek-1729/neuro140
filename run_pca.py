import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

folders = os.listdir("MAEC_Dataset")


for i in tqdm(range(len(folders))):
    try:
        folder = folders[i]
        path = f"MAEC_Dataset/{folder}"
        # Read the CSV file
        data = pd.read_csv(f'{path}/features.csv')
        df = pd.read_csv("change.csv")
        label = list(df[df['Folder'] == folder]['Label'])[0]

        data_numeric = data.apply(pd.to_numeric, errors='coerce').dropna()
        # Apply PCA to all columns
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(data_numeric)

        # Create a DataFrame for the PCA result
        pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])
        pca_df['Label'] = label
        pca_df.to_csv(f'{path}/pca.csv', index=False)
    except:
        print(f'{folder} did not work. Iteration {i}')
# Plot the PCA result
# plt.figure(figsize=(8, 6))
# plt.scatter(pca_df['PCA1'], pca_df['PCA2'], alpha=0.5)
# plt.title('PCA of features.csv')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.grid(True)
# plt.show()
