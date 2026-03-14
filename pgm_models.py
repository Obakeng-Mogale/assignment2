import matplotlib.pyplot as plt
import numpy as np


class GaussianNB:
    """assume X is dxN"""
    def __init__(self):
        self.class_means = {}
        self.prior_class_probabilities = {}
        self.data = None
        self.cov = {}
        self.response = None
        self.N = None
        self.d = None
        self.log_odds = None
        self.odds = None
        return

    def fit(self, X, y):
        self.response = y
        self.d, self.N = X.shape
        self.data = X
        self.get_class_means()
        self.get_cov_mat()
        self.priors_calc()
        return

    def predict(self, X):
        probs = self.get_probs(X)
        # Find the index of the max probability per sample
        class_indices = np.argmax(probs, axis=0)
        classes = np.unique(self.response)
        return classes[class_indices] 

    def priors_calc(self):
        # New: Calculate priors
        for label in np.unique(self.response):
            self.prior_class_probabilities[label] = np.sum(label == self.response) / self.N
        return self.prior_class_probabilities

    def get_class_means(self):
        for label in np.unique(self.response):
            # Select columns where label matches and average across axis 1 (samples)
            self.class_means[label] = np.mean(self.data[:, label == self.response], axis=1).reshape(-1, 1)
        return self.class_means

    def get_cov_mat(self):
        diag_sigma = np.zeros(self.d)
        i = 0
        for label in np.unique(self.response):
            xnj = self.data[:,label == self.response]
            nj = xnj.shape[1]
            sigma2 = np.sum((xnj- self.class_means[label])**2, axis = 1 )
            sigma2 = sigma2/nj
            print(sigma2)
            self.cov[label] = sigma2
            i+=1 
        return self.cov

    def gaussianpdf(self, x):
        n_classes = np.unique(self.response).shape[0]
        n_samples = x.shape[1]
        # pred should be (n_classes, n_samples)
        pred = np.zeros((n_classes, n_samples))
    
        # Loop through each individual test sample
        for sample_idx in range(n_samples):
            # Extract the specific d-dimensional vector for this sample
            current_x = x[:, sample_idx].reshape(-1, 1) 
        
            # Loop through each class to get the likelihood for this sample
            for class_idx, label in enumerate(np.unique(self.response)):
                """coefficient""" 
                det_val = np.linalg.det(np.diag(2 * np.pi * self.cov[label]))
                a = 1 / np.sqrt(det_val)
            
                """
                diff = x-muj
                inv_cov = sigma^-1
                """
                diff = current_x - self.class_means[label].reshape(-1, 1)
                inv_cov = np.linalg.inv(np.diag(self.cov[label]))
            
                # Use .item() to avoid the "sequence" ValueError from before
                exponent = -0.5 * (diff.T @ inv_cov @ diff)
                b = np.exp(exponent.item())
            
                # Store likelihood for this class and this sample
                pred[class_idx, sample_idx] = a * b

        return pred

    def get_probs(self, X):
        # 1. Get likelihoods p(x|Cj) -> shape (n_classes, n_samples)
        likelihoods = self.gaussianpdf(X)

        posteriors = np.zeros_like(likelihoods)
        for i,label in enumerate(np.unique(self.response)):
            posteriors[i,:] = likelihoods[i,:]*self.prior_class_probabilities[label]
            
        
        return posteriors/np.sum(posteriors,axis=0)

    def get_odds(self, X):
        probs = self.get_probs(X)
        n_classes = np.unique(self.response).shape[0]
        self.odds = np.zeros((n_classes,n_samples))
        for x in range(n_classes):
            self.odds[i] = probs[i,:] / 1-probs[i,:] 
        return self.odds

    def get_log_odds(self, X):
        self.log_odds = np.log(self.get_odds(X))
        return self.log_odds



""" LogisticRegression implementation"""
class LogisticRegression:

    """
    2 class case
    """
    def __init__(self, lambda_ = 1.0):
        self.data = None
        self.response = None
        self.class_means = {}
        self.d = None
        self.n = None
        self.gradient = None
        self.lambda_param = None
        self.lambda_ = lambda_
        self.H = None
        return

    def fit(self, X, y, iterations =10, tolerance = 0.001):
        self.d , self.n = X.shape
        # adding the 1s columns
        one_vec = np.ones(self.n)
        self.w = np.zeros(self.d+1)
        self.response = y

        """contains the w vectors"""
        self.data = np.vstack([one_vec, X])

        """ newton_raphson loop"""
        for i in range(iterations):
            self.newton_raphson()
        return

    def predict(self, X):
        n = X.shape[1]
        one_vec = np.ones(n)
        X = np.vstack([one_vec, X])
        a = self.w.T @ X
        prob = 1/(1+np.exp(-a))
        return np.where(prob>0.5,1,0)

    def sigmoid(self):
        a = self.w.T @ self.data

        return  1 / (1 + np.exp(-a))   
    def get_gradient(self):
        self.gradient = self.data @ (self.sigmoid() - self.response ) + (1 /self.lambda_) * self.w
        return self.gradient

    def get_Hessian(self):
        self.H = self.data @ np.diag(self.sigmoid() * (1-self.sigmoid()))  @ self.data.T +(1/self.lambda_) * np.eye(self.d + 1)
        return self.H

    def newton_raphson(self):
        self.w = self.w - np.linalg.inv(self.get_Hessian()) @ self.get_gradient()
        return 
""" Multiple LogisticRegression class"""
class Softmax_Classifier:
    def __init__(self, lr=1):
        self.d = None
        self.n = None
        self.W = None
        self.one_hot = None # also known as tn
        self.lr = lr
        return 

    def fit(self, X, y): 
        self.d,self.n = X.shape
        self.one_hot = np.eye(np.unique(y).shape[0],dtype = int)[y].T
        one_vec = np.ones(self.n)
        self.data = np.vstack([one_vec, X])
        #initialize W 
        self.W = np.zeros((self.d+1, np.unique(y).shape[0]))
        for i in range(1):
            self.gradient_descent()
        return

    def predict(self, X):
        n = X.shape[1]
        one_vec = np.ones(n)
        X = np.vstack([one_vec, X])
        a = self.W.T @ X
        exp_a = np.exp(a)
        prob =  exp_a/np.sum(exp_a, axis = 0)
        return np.argmax(prob, axis = 0)

    def sigmoid_softmax(self):
        a = self.W.T @ self.data
        exp_a = np.exp(a)
        return exp_a/np.sum(exp_a, axis = 0) 

    def get_gradient(self):
# 1. The pure error matrix: Y - T (shape: k x n)
        error = self.sigmoid_softmax() - self.one_hot
        
        # 2. Matrix multiplication handles the summation over all n samples!
        # self.data is (d+1, n), error.T is (n, k) -> result is (d+1, k)
        self.gradient = self.data @ error.T
        return self.gradient 

    def gradient_descent(self):
        self.W = self.W - self.lr * self.get_gradient()
        return 
    
if __name__ == "__main__":
# Set a random seed so you get the same dataset every time you run it
    np.random.seed(42)

# Define the number of data points per class
    N = 100

# ---------------------------------------------------------
# Create Class 0
# Centered around coordinates (2, 2)
# ---------------------------------------------------------
    mean_0 = [2, 2]
    cov_0 = [[1, 0.5], 
            [0.5, 1]]  # Covariance matrix dictates the "shape" of the cluster
    X_0 = np.random.multivariate_normal(mean_0, cov_0, N)
    y_0 = np.zeros(N)   # Labels for Class 0 are all 0s


# ---------------------------------------------------------
# Create Class 1
# Centered around coordinates (-2, -2)
# ---------------------------------------------------------
    mean_1 = [-2, -2]
    cov_1 = [[1, 0.5], 
            [0.5, 1]]
    X_1 = np.random.multivariate_normal(mean_1, cov_1, N)
    y_1 = np.ones(N)    # Labels for Class 1 are all 1s

# ---------------------------------------------------------
# Combine and Shuffle
# ---------------------------------------------------------
    X = np.vstack((X_0, X_1)) # Shape becomes (200, 2)
    y = np.hstack((y_0, y_1)) # Shape becomes (200,)

# It's good practice to shuffle your data so the model 
# doesn't see all 0s followed by all 1s
    shuffle_idx = np.random.permutation(2 * N)
    X = X[shuffle_idx]
    y = y[shuffle_idx]

# ---------------------------------------------------------
# Visualize the Dataset
# ---------------------------------------------------------
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 0', alpha=0.7)
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1', alpha=0.7)
    plt.title("Synthetic 2-Class Dataset for Logistic Regression")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True)
    plt.show()
