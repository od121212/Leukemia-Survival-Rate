import numpy as np
import pandas as pd
from scipy.stats import jarque_bera
import matplotlib.pyplot as plt
import seaborn as sns


class dtmanager:

    def __init__(
            self, 
            df:pd.DataFrame, 
            target:pd.DataFrame
    ):
        
        self.df = df
        self.y = target
        self.float_cols = self.df.select_dtypes(include=['float64']).columns
    
    
    def stats_analysis(self)->pd.DataFrame:

        stats = self.df.describe(include='all').transpose()
        stats["dtype"] = self.df.dtypes
        stats["distinct_values"] = self.df.nunique()
        stats["count_nan"] = self.df.isna().sum()
        stats["nan_percentage"] = (self.df.isna().sum() / len(self.df)) * 100

        cols_order = ['dtype', 'count_nan', 'nan_percentage', 'distinct_values'] 
        remaining_cols = [c for c in stats.columns if c not in cols_order]

        final = stats[cols_order + remaining_cols]
        
        return final.drop(columns=['unique', 'top', 'freq'])
    

    def target_analysis(self):

        stats_target = self.y.describe().transpose()
        
        return stats_target
    

    def float_columns_normality_test(self)->pd.DataFrame:
        
        normality_results = {}

        for col in self.float_cols:
            stat, p_value = jarque_bera(self.df[col].dropna())
            normality_results[col] = {
                'statistic': stat,
                'p_value': p_value,
                'is_normal': p_value > 0.05
            }

        return pd.DataFrame(normality_results).transpose()
    

    def plot_float_distributions(self):

        fig, axes = plt.subplots(len(self.float_cols), 1, figsize=(10, 5 * len(self.float_cols)))
        if len(self.float_cols) == 1:
            axes = [axes]

        for i, col in enumerate(self.float_cols):
            sns.histplot(self.df[col].dropna(), kde=True, ax=axes[i], color='skyblue')
            axes[i].set_title(f'Distribution of {col}', fontsize=14)
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')

        plt.tight_layout()
        plt.show()


    def plot_target_distribution(self):

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        sns.histplot(self.y['OS_YEARS'], kde=True, ax=axes[0], color='salmon')
        axes[0].set_title('Distribution life survival duration (OS_YEARS)')

        sns.countplot(x='OS_STATUS', data=self.y, ax=axes[1], palette='Set2')
        axes[1].set_title('Distribution of survival status (OS_STATUS)')

        plt.tight_layout()
        plt.show()


    def plot_correlation_matrix(self):

        plt.figure(figsize=(12, 8))

        corr = self.df.select_dtypes(include=[np.number]).corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title("Correlation Matrix of Numerical Features")
        plt.show()

    
    def plot_distribution_by_target(self):

        temp_df = self.df.merge(self.y[['OS_YEARS', 'OS_STATUS']], left_index=True, right_index=True)

        fig, axes = plt.subplots(len(self.float_cols), 1, figsize=(10, 5 * len(self.float_cols)))

        for i, col in enumerate(self.float_cols):
            sns.scatterplot(data=temp_df, x='OS_YEARS', y=col, hue='OS_STATUS', ax=axes[i], alpha=0.5)
            axes[i].set_title(f'{col} vs Years of Survival', fontsize=14)

        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    df = pd.read_csv("./X_train/clinical_train.csv", index_col=0)
    target_df = pd.read_csv("./target_train.csv", index_col=0)
    dtm = dtmanager(df, target_df)
    print(dtm.plot_target_distribution())