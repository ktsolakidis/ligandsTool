import numpy as np
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt

def mahalanobis_plot(dataframe, column_names, threshold):
    if not all(column in dataframe.columns for column in column_names):
        raise ValueError("One or more specified columns do not exist in the dataframe")

    if 'Ligand' not in dataframe.columns:
        raise ValueError("'Ligand' column not found in the dataframe")

    num_columns = len(column_names)
    if num_columns % 2 != 0 or num_columns > 8:
        raise ValueError("Number of columns must be even and less than or equal to 8")

    num_plots = num_columns // 2
    num_rows = 2 if num_plots > 2 else 1
    num_cols = min(2, num_plots)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, num_rows * 5))

    if num_plots == 1:
        axes = [axes]

    if num_plots > 2:
        axes = axes.flatten()

    outliers_data_list = []

    for i in range(num_plots):
        ax = axes[i]
        start_idx = i * 2
        end_idx = start_idx + 2
        subset_columns = column_names[start_idx:end_idx]
        df_subset = dataframe[['Ligand'] + subset_columns].dropna()

        try:
            cov_matrix = np.cov(df_subset[subset_columns], rowvar=False)
            inv_cov_matrix = np.linalg.inv(cov_matrix)
            mean = df_subset[subset_columns].mean().values
            mahalanobis_distances = [mahalanobis(row, mean, inv_cov_matrix) for row in df_subset[subset_columns].values]

            is_outlier = np.array(mahalanobis_distances) > threshold
            outliersData = df_subset[is_outlier]  # Outliers including 'ligand'
            outliers_data_list.append(outliersData)

            ax.scatter(df_subset[subset_columns[0]], df_subset[subset_columns[1]], label='Inliers', color='blue')
            ax.scatter(outliersData[subset_columns[0]], outliersData[subset_columns[1]], label='Outliers', color='red')
            ax.set_xlabel(subset_columns[0])
            ax.set_ylabel(subset_columns[1])
            ax.legend()
        except Exception as e:
            print(f"Error for columns {subset_columns[0]} and {subset_columns[1]}: {e}")

    plt.tight_layout()
    plt.show()

    return outliers_data_list
