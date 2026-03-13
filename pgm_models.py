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
    def __init__(self):
        self.data = None
        self.response = None
        self.class_means = {}
        self.prior_class_probabilities = None
        self.class_variances = None
        self.d = None
        self.n = None
        self.gradient = None
        return
    def fit(self, X, y):
        self.d , self.n = X.shape
        # adding the 1s columns
        one_vec = np.ones(self.n)
        self.w = np.zeros(self.n+1)
        self.response = y

        """contains the w vectors"""
        self.data = np.vstack([X, one_vec])
        return

    def predict(self, X):
        return

    def get_cov(self):
        return 

    def sigmoid(self):
        return 1/(1+np.exp(-(self.w.T @ self.data @)))
    
    def get_gradient(self):
        for n in range(self.n):
            self.gradient += (self.sigmoid()[:,n] - self.response ) @
                self.data[:,n] + 1/self.lambda * self.w
        return 

    def get_Hessian(self):
        for n in range(self.n):
            self.H = self.sigmoid()[:,n]@(1-self.sigmoid()[:,n]) @
                self.data[:,n]@ self.data[:,n].T + 1/self.lambda * np.eye(self.n)
        return

    def newton_raphson(self):
        return 

    
if __name__ == "__main__":
    model = GaussianNB()
    X = np.array([
        [-1.1, -0.9], # C1 sample 1
        [-0.9, -1.1], # C1 sample 2
        [-1.0, -1.0], # C1 sample 3
        [ 0.9,  1.1], # C2 sample 1
        [ 1.1,  0.9], # C2 sample 2
        [ 1.0,  1.0]  # C2 sample 3
        ])

    X = X.T
    y = np.array([0, 0, 0, 1, 1, 1])
    
    X_test = np.array([
    [-1.0, -1.0],  # Case A: Exactly on Mean C0
    [ 1.0,  1.0],  # Case B: Exactly on Mean C1
    [-0.5, -0.5],  # Case C: Deep in C0 territory
    [ 0.0,  0.0],  # Case D: The exact Decision Boundary
    [ 2.0,  2.0]   # Case E: Far out in C1 territory
    ])
    X_test = X_test.T
    model.fit(X, y)
    print(model.class_means)
    print(model.cov)
    print(model.predict(X_test))
