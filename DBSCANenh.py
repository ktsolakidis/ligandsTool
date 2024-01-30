import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def perform_dbscan_clustering(dataframe, column_names, eps, min_samples):
    if not all(column in dataframe.columns for column in column_names):
        raise ValueError("One or more specified columns do not exist in the dataframe")

    if 'Ligand' not in dataframe.columns:
        raise ValueError("'Ligand' column not found in the dataframe")

    num_columns = len(column_names)
    if num_columns % 2 != 0 or num_columns > 8:
        raise ValueError("Number of columns must be 2, 4, 6, or 8")

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
            db = DBSCAN(eps=eps, min_samples=min_samples)
            clusters = db.fit_predict(df_subset[subset_columns])
          
            outliersData = df_subset[clusters == -1]  # Outliers data including 'ligand'
            outliers_data_list.append(outliersData)

            ax.scatter(df_subset[subset_columns[0]], df_subset[subset_columns[1]], c=clusters)
            ax.set_xlabel(subset_columns[0])
            ax.set_ylabel(subset_columns[1])
        except Exception as e:
            print(f"Error in clustering for columns {subset_columns[0]} and {subset_columns[1]}: {e}")

    plt.tight_layout()
    plt.show()

    return outliers_data_list
