"""
Stock Market Analysis Tool

This module provides a comprehensive set of tools for analyzing stock market data,
including covariance matrix estimation, eigenvalue analysis, and spectral clustering.

Author: Mukul Chodhary (1172562)
Date: September 1, 2024
"""

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, probplot
from sklearn.covariance import shrunk_covariance, ledoit_wolf, oas
from sklearn.cluster import SpectralClustering

class StockAnalysis:
    """
    A class for analyzing stock market data using various statistical methods.

    This class provides functionality to load stock return data and metadata,
    calculate covariance matrices using different methods, perform spectral
    clustering, and visualize the results.

    Attributes:
        returns_data (pd.DataFrame): Daily stock returns data.
        metadata (pd.DataFrame): Metadata for the stocks.
        cov_matrix (np.ndarray): Covariance matrix of stock returns.
        eigenvalues (np.ndarray): Eigenvalues of the covariance matrix.
        eigenvectors (np.ndarray): Eigenvectors of the covariance matrix.
    """

    def __init__(self, returns_file_path: str, metadata_file_path: str) -> None:
        """
        Initialize the StockAnalysis object.

        Args:
            returns_file_path (str): Path to the CSV file containing stock returns.
            metadata_file_path (str): Path to the CSV file containing stock metadata.
        """
        self.returns_data = pd.read_csv(returns_file_path, header=None)
        # normalise
        self.returns_data = (self.returns_data - self.returns_data.mean()) / self.returns_data.std()
        self.metadata = pd.read_csv(metadata_file_path)
        self.cov_methods = {
            'Sample Covariance Matrix': self.scm,
            'Linear Shrinkage': self.linear_shrinkage,
            'Ledoit-Wolf Shrinkage': self.ledoit_wolf_shrinkage,
            'Oracle Approximating Shrinkage': self.oracle_approximating_shrinkage
        }
        self.cov_method = None
        self.cov_matrix: np.ndarray = None
        self.eigenvalues: np.ndarray = None
        self.eigenvectors: np.ndarray = None

    def scm(self) -> np.ndarray:
        """
        Calculate and return the sample covariance matrix of stock returns.

        Returns:
            np.ndarray: The calculated covariance matrix.
        """
        return np.cov(self.returns_data)

    def linear_shrinkage(self, alpha: float = 0.1) -> np.ndarray:
        """
        Calculate the linear shrinkage estimate of the covariance matrix.

        Args:
            alpha (float): Shrinkage intensity, default is 0.1.

        Returns:
            np.ndarray: The linear shrinkage estimate of the covariance matrix.
        """
        sample_cov = np.cov(self.returns_data)
        target_cov = np.diag(np.diag(sample_cov))
        return (1 - alpha) * sample_cov + alpha * target_cov

    def ledoit_wolf_shrinkage(self) -> np.ndarray:
        """
        Calculate the Ledoit-Wolf shrinkage estimate of the covariance matrix.

        Returns:
            np.ndarray: The Ledoit-Wolf shrinkage estimate of the covariance matrix.
        """
        return ledoit_wolf(self.returns_data.T)[0]

    def oracle_approximating_shrinkage(self) -> np.ndarray:
        """
        Calculate the Oracle Approximating Shrinkage estimate of the covariance matrix.

        Returns:
            np.ndarray: The Oracle Approximating Shrinkage estimate of the covariance matrix.
        """
        return oas(self.returns_data.T)[0]

    def generate_scree_plot(self) -> None:
        """
        Generate and display a scree plot of eigenvalues.
        """
        if self.eigenvalues is None:
            self.eigenvalues, _ = np.linalg.eig(self.cov_matrix)
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(self.eigenvalues) + 1), self.eigenvalues, marker='o')
        plt.title('Scree Plot')
        plt.xlabel('Eigenvalue Index')
        plt.ylabel('Eigenvalue')
        plt.grid()
        plt.show()

    def eigenvalue_histogram(self) -> None:
        """
        Generate and display a histogram of eigenvalues.
        """
        if self.eigenvalues is None:
            self.eigenvalues, _ = np.linalg.eig(self.cov_matrix)
        plt.figure(figsize=(8, 5))
        plt.hist(self.eigenvalues, density=True, bins=30, edgecolor='black', alpha=0.7)
        plt.title('Histogram of Eigenvalues')
        plt.xlabel('Eigenvalue')
        plt.ylabel('Density')
        plt.grid()
        plt.show()

    # def compare_with_gaussian(self) -> None:
    #     """
    #     Compare the eigenvalue distribution with that of a Gaussian random matrix using side-by-side plots.
    #     """
    #     n, t = self.returns_data.shape
    #     gaussian_matrix = np.random.randn(n, t)
    #     gaussian_cov = np.cov(gaussian_matrix)
    #     gaussian_eigenvalues, _ = np.linalg.eig(gaussian_cov)

        
    #     # Sort eigenvalues in descending order and exclude the largest one
    #     sorted_eigenvalues = np.sort(self.eigenvalues)[::-1][1:]

    #     fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    #     axes[0].hist(sorted_eigenvalues, density=True, bins=50, alpha=0.7, edgecolor='black')
    #     axes[0].set_title('Eigenvalue Distribution (Stock Data)')
    #     axes[0].set_xlabel('Eigenvalue')
    #     axes[0].set_ylabel('Density')
    #     axes[0].grid(True)

    #     axes[1].hist(gaussian_eigenvalues, density=True, bins=50, alpha=0.7, edgecolor='black')
    #     axes[1].set_title('Eigenvalue Distribution (Gaussian Random Matrix)')
    #     axes[1].set_xlabel('Eigenvalue')
    #     axes[1].set_ylabel('Density')
    #     axes[1].grid(True)

    #     plt.tight_layout()
    #     plt.show()

    def compare_with_gaussian(self) -> None:
        """
        Compare the eigenvalue distribution with that of a Gaussian random matrix.
        """
        n, t = self.returns_data.shape
        gaussian_matrix = np.random.randn(n, t)
        gaussian_matrix = (gaussian_matrix - np.mean(gaussian_matrix)) / np.std(gaussian_matrix)
        gaussian_cov = self.cov_method(gaussian_matrix)
        gaussian_eigenvalues, _ = np.linalg.eig(gaussian_cov)
    
        plt.figure(figsize=(10, 5))
        plt.hist(self.eigenvalues, density=True, bins=25, alpha=0.5, label='Stock Data')
        plt.hist(gaussian_eigenvalues,density=True, bins=20, alpha=0.5, label='Gaussian Random Matrix')
        plt.title('Comparison of Eigenvalue Distributions')
        plt.xlabel('Eigenvalue')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid()
        plt.show()

    def analyze_top_eigenvectors(self, k=10) -> None:
        """
        Analyze and visualize the top eigenvectors of the covariance matrix.
        """
        self.calculate_eigen_decomposition()
        top_indices = np.argsort(self.eigenvalues)[-k:][::-1]  # Top 5 eigenvalues
        top_eigenvectors = self.eigenvectors[:, top_indices]
        top_eigenvalues = self.eigenvalues[top_indices]

        fig, axes = plt.subplots(k, 3, figsize=(20, 25))
        fig.suptitle('Analysis of Top 5 Eigenvectors', fontsize=16)

        for i in range(k):
            # Histogram
            axes[i, 0].hist(top_eigenvectors[:, i], density=True, bins=30, alpha=0.7)
            axes[i, 0].set_title(f'Histogram of Eigenvector {i+1}')
            axes[i, 0].set_xlabel('Value')
            axes[i, 0].set_ylabel('Density')
            axes[i, 0].grid(True)

            # Q-Q Plot
            probplot(top_eigenvectors[:, i], dist="norm", plot=axes[i, 1])
            axes[i, 1].set_title(f'Q-Q Plot of Eigenvector {i+1}')
            axes[i, 1].grid(True)

            # Bar Plot of eigenvector entries
            axes[i, 2].bar(range(len(top_eigenvectors[:, i])), top_eigenvectors[:, i], alpha=0.7)
            axes[i, 2].set_title(f'Bar Plot of Eigenvector {i+1} Entries')
            axes[i, 2].set_xlabel('Entry Index')
            axes[i, 2].set_ylabel('Entry Value')
            axes[i, 2].grid(True)
            # Add eigenvalue information as text
            axes[i, 2].text(0.95, 0.95, f'Eigenvalue: {top_eigenvalues[i]:.2e}', 
                        verticalalignment='top', horizontalalignment='right',
                        transform=axes[i, 2].transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))


        plt.tight_layout()
        plt.show()

    def covariance_matrix_reconstruction(self) -> np.ndarray:
        """
        Reconstruct the covariance matrix using top eigenvalues and eigenvectors.

        Returns:
            np.ndarray: The reconstructed covariance matrix.
        """
        self.calculate_eigen_decomposition()
        top_indices = np.argsort(self.eigenvalues)[-5:][::-1]
        top_eigenvalues = self.eigenvalues[top_indices]
        top_eigenvectors = self.eigenvectors[:, top_indices]

        reconstructed_cov = (top_eigenvectors @ np.diag(top_eigenvalues) @ top_eigenvectors.T)
        return reconstructed_cov

    def calculate_eigen_decomposition(self) -> None:
        """
        Calculate the eigenvalues and eigenvectors of the covariance matrix.
        """
        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.cov_matrix)

    def spectral_clustering(self, n_clusters: int = 5) -> None:
        """
        Perform spectral clustering on the stock data and visualize the results.

        Args:
            n_clusters (int): Number of clusters to form, default is 5.
        """
        clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
        similarity_matrix = np.abs(np.corrcoef(self.returns_data))
        labels = clustering.fit_predict(similarity_matrix)

        results = pd.DataFrame({
            'Symbol': self.metadata['Symbol'],
            'Company Name': self.metadata['Company Name'],
            'Sector': self.metadata['Sector'],
            'Cluster': labels
        })

        for cluster in range(n_clusters):
            print(f"\nCluster {cluster}:")
            cluster_stocks = results[results['Cluster'] == cluster]
            print(cluster_stocks[['Symbol', 'Company Name', 'Sector']])

        plt.figure(figsize=(12, 8))
        for cluster in range(n_clusters):
            cluster_stocks = results[results['Cluster'] == cluster]
            plt.scatter(cluster_stocks.index, [cluster] * len(cluster_stocks), label=f'Cluster {cluster}')
        plt.yticks(range(n_clusters))
        plt.xlabel('Stock Index')
        plt.ylabel('Cluster')
        plt.title('Spectral Clustering of Stocks')
        plt.legend()
        plt.show()

    def compare_covariance_estimators(self) -> None:
        """
        Compare different covariance matrix estimation methods and visualize the results.
        """
        estimators: List[Tuple[str, np.ndarray]] = [
            ("Sample Covariance Matrix", self.scm()),
            ("Linear Shrinkage", self.linear_shrinkage()),
            ("Ledoit-Wolf Shrinkage", self.ledoit_wolf_shrinkage()),
            ("Oracle Approximating Shrinkage", self.oracle_approximating_shrinkage())
        ]

        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle("Comparison of Covariance Matrix Estimators")

        for (title, estimator), ax in zip(estimators, axes.ravel()):
            im = ax.imshow(estimator, cmap='coolwarm')
            ax.set_title(title)
            plt.colorbar(im, ax=ax)

        plt.tight_layout()
        plt.show()

        print("Condition Numbers:")
        for title, estimator in estimators:
            cond_num = np.linalg.cond(estimator)
            print(f"{title}: {cond_num:.2f}")

        print("\nFrobenius Norms (difference from SCM):")
        for title, estimator in estimators[1:]:
            frob_norm = np.linalg.norm(estimator - estimators[0][1], 'fro')
            print(f"{title}: {frob_norm:.2f}")

    def complete_analysis(self, cov_method: str = 'Sample Covariance Matrix') -> None:
        """
        Perform a complete analysis of the stock data using the specified covariance method.

        Args:
            cov_method (str): The covariance estimation method to use. 
                              Options: 'Sample Covariance Matrix', 'Linear Shrinkage', 
                              'Ledoit-Wolf Shrinkage', 'Oracle Approximating Shrinkage'
        """
        if cov_method not in self.cov_methods:
            raise ValueError(f"Invalid covariance method. Choose from: {', '.join(self.cov_methods.keys())}")
        
        self.cov_method = cov_method
        self.cov_matrix = self.cov_methods[cov_method]()
        self.generate_scree_plot()
        self.eigenvalue_histogram()
        self.compare_with_gaussian()
        self.analyze_top_eigenvectors()
        self.covariance_matrix_reconstruction()
        self.spectral_clustering()