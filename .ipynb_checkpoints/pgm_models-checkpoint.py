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
        self.update_mat = None
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
            #print(sigma2)
            self.cov[label] = sigma2
            i+=1 
        return self.cov
    def gaussianpdfs(self, x):
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
    def gaussianpdf(self, x):
        n_classes = len(np.unique(self.response))
        n_samples = x.shape[1]
        pred = np.zeros((n_classes, n_samples))
        
        # We still loop over classes (usually a small number like 2 to 10), 
        # but we process ALL samples for that class at once!
        for class_idx, label in enumerate(np.unique(self.response)):
            
            # mu shape: (d, 1)
            mu = self.class_means[label] 
            
            # var shape: (d, 1) - Extract the diagonal variances
            var = self.cov[label].reshape(-1, 1) 
            
            # 1. OPTIMIZE DETERMINANT: 
            # The determinant of a diagonal matrix is just the product of its diagonal elements.
            # det(2 * pi * Cov) = Product(2 * pi * variance_i)
            det_val = np.prod(2 * np.pi * var)
            a = 1.0 / np.sqrt(det_val)
            
            # 2. BROADCASTING:
            # x is shape (d, N), mu is shape (d, 1). 
            # NumPy automatically subtracts mu from EVERY column in x simultaneously!
            diff = x - mu 
            
            # 3. OPTIMIZE MATRIX INVERSE & MULTIPLICATION:
            # For a diagonal matrix, (x - mu)^T * Cov^-1 * (x - mu) strictly simplifies 
            # to the sum of ((x - mu)^2 / variance) across all features.
            # diff**2 / var calculates this for ALL samples simultaneously yielding shape (d, N).
            # np.sum(..., axis=0) sums down the columns, yielding an array of shape (N,)
            exponent = -0.5 * np.sum((diff ** 2) / var, axis=0)
            
            # 4. CALCULATE EXPONENTIAL FOR ALL SAMPLES
            b = np.exp(exponent)
            
            # Store the likelihoods for this class (shape N,)
            pred[class_idx, :] = a * b

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
        n_classes, n_samples = probs.shape
       
        self.odds = np.zeros((n_classes,n_samples))
        for i in range(n_classes):
            #self.odds[i, :] = probs[i,:] / (1 - probs[i,:] + 1e-15)
            self.odds[i] = probs[i,:] / (1-probs[i,:]) 
        return self.odds

    def get_log_odds(self, X):
       # self.log_odds = np.log(self.get_odds(X) + 1e-15)
        self.log_odds = np.log(self.get_odds(X))
        return self.log_odds
    def predict_log_proba(self, X):
        """Returns the log of the posterior probabilities."""
        # Add a tiny epsilon to prevent log(0)
   
        probs = self.get_probs(X)
        """uncommoent if div0 err"""
        #return np.log(probs + 1e-15)
        return np.log(probs)

    def get_joint_log_likelihood(self, X):
        """Calculates the unnormalized log-posterior: ln(P(x|C)) + ln(P(C))."""
        likelihoods = self.gaussianpdf(X)
        n_classes = len(np.unique(self.response))
        n_samples = X.shape[1]
        
        joint_log_lik = np.zeros((n_classes, n_samples))
        for i, label in enumerate(np.unique(self.response)):
            # Add epsilon to prevent log(0)
            log_lik = np.log(likelihoods[i, :] + 1e-15)
            log_prior = np.log(self.prior_class_probabilities[label])
            joint_log_lik[i, :] = log_lik + log_prior
            
        return joint_log_lik

    def get_marginal_log_likelihood(self, X):
        """
        Calculates the log of the evidence/denominator using the Log-Sum-Exp trick.
        p(x) = sum(p(x|C) * p(C))
        """
        joint_log_lik = self.get_joint_log_likelihood(X)
        
        # Log-Sum-Exp trick for numerical stability
        max_log = np.max(joint_log_lik, axis=0)
        # Subtract max before exp, then add it back outside the log
        marginal = max_log + np.log(np.sum(np.exp(joint_log_lik - max_log), axis=0))
        
        return marginal

    def score(self, X, y):
        """Returns the accuracy of the model."""
        predictions = self.predict(X)
        return np.mean(predictions == y)


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
    def get_log_odds(self, X):
        """
        In Binary Logistic Regression, the log-odds is mathematically 
        identical to the linear activation a = w^T * x.
        """
        n = X.shape[1]
        one_vec = np.ones(n)
        X_aug = np.vstack([one_vec, X])
        
        # We don't need to run it through the sigmoid, just return the linear part!
        log_odds = self.w.T @ X_aug
        return log_odds
    def get_loss(self):
        """Calculates the negative log-posterior."""
        preds = self.sigmoid()
        # Add a tiny epsilon (e.g., 1e-15) inside logs to prevent log(0) errors
        eps = 1e-15 
        log_likelihood = np.sum(self.response * np.log(preds + eps) + 
                               (1 - self.response) * np.log(1 - preds + eps))
        
        # Include the L2 regularization penalty term (1/2 * lambda * w^T * w)
        penalty = (1 / (2 * self.lambda_)) * np.sum(self.w**2)
        
        return -log_likelihood + penalty

    def get_bias(self):
        """Returns the bias term (w_0)."""
        return self.w[0] # Use self.W[0, :] for Softmax
    def score(self, X, y):
        """Returns the accuracy of the model."""
        predictions = self.predict(X)
        return np.mean(predictions == y)
    def get_weights(self):
        """Returns the feature weights (w_1 to w_d)."""
        return self.w[1:] # Use self.W[1:, :] for Softmax

        
    def predict_proba(self, X):
        """Returns the raw probabilities for the input data."""
        n = X.shape[1]
        one_vec = np.ones(n)
        X_aug = np.vstack([one_vec, X])
        
        a = self.w.T @ X_aug # Use self.W.T for Softmax_Classifier
        
        # For 2-class:
        
        return 1 / (1 + np.exp(-a)) 
        
        # For Softmax_Classifier use:
        # exp_a = np.exp(a)
        # return exp_a / np.sum(exp_a, axis=0)`

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
        self.update_mat = np.linalg.inv(self.get_Hessian()) @ self.get_gradient()
        self.w = self.w - np.linalg.inv(self.get_Hessian()) @ self.get_gradient()
        return 
    def get_weighting_matrix(self):
        """
        Returns the NxN diagonal weighting matrix S.
        The diagonal contains the variance of the predictions: sigma * (1 - sigma).
        """
        if self.data is None:
            raise ValueError("Model must be fitted to extract the weighting matrix.")
            
        preds = self.sigmoid()
        # Create a diagonal matrix from the 1D array of variances
        S = np.diag(preds * (1 - preds))
        return S

    def get_update_vector(self):
        """
        Returns the Delta w step size (H^-1 * Gradient) based on current weights.
        """
        if self.data is None:
            raise ValueError("Model must be fitted to extract the update vector.")
            
        H_inv = np.linalg.inv(self.get_Hessian())
        grad = self.get_gradient()
        
        delta_w = H_inv @ grad
        return delta_w

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
    from sklearn.datasets import make_classification
    from sklearn.naive_bayes import GaussianNB as SklearnGaussianNB
    from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
    import numpy as np

    # ==========================================
    # 0. Generate Random 2-Class Data
    # ==========================================
    print("Generating random dataset (100 samples, 4 features)...")
    X, y = make_classification(n_samples=100, n_features=4, n_classes=2, 
                               n_informative=2, random_state=42)

    # ==========================================
    # 1. Test Gaussian Naive Bayes
    # ==========================================
    print("\n" + "="*50)
    print("TESTING GAUSSIAN NAIVE BAYES")
    print("="*50)

    # Fit Custom Model
    custom_gnb = GaussianNB()
    custom_gnb.fit(X.T, y)

    # Fit Sklearn Model
    sk_gnb = SklearnGaussianNB()
    sk_gnb.fit(X, y)

    # --- Compare Means ---
    # Sklearn stores means in shape (k, d). Custom stores as a dict of (d, 1) vectors.
    custom_means = np.vstack([custom_gnb.class_means[0].T, custom_gnb.class_means[1].T])
    print("\nClass Means Match:")
    print(np.allclose(custom_means, sk_gnb.theta_))

    # --- Compare Predictions ---
    custom_preds = custom_gnb.predict(X.T)
    sk_preds = sk_gnb.predict(X)
    print("Predictions Match exactly:")
    print(np.array_equal(custom_preds, sk_preds))

    # --- Compare Probabilities ---
    # Custom get_probs returns (k, N). Sklearn returns (N, k). We must transpose!
    custom_probs = custom_gnb.get_probs(X.T).T
    sk_probs = sk_gnb.predict_proba(X)
    
    # We use allclose because floating point math might differ by 0.0000000001
    print("Probabilities Match (within tolerance):")
    print(np.allclose(custom_probs, sk_probs, atol=1e-5))

    # --- Compare Log-Probabilities ---
    custom_log_probs = custom_gnb.predict_log_proba(X.T).T
    sk_log_probs = sk_gnb.predict_log_proba(X)
    print("Log-Probabilities Match (within tolerance):")
    print(np.allclose(custom_log_probs, sk_log_probs, atol=1e-5))


    # ==========================================
    # 2. Test Binary Logistic Regression
    # ==========================================
    print("\n" + "="*50)
    print("TESTING BINARY LOGISTIC REGRESSION")
    print("="*50)

    # Fit Custom Model
    # We use a massive lambda (1e9) to effectively turn off regularization 
    # so we can perfectly match Sklearn's unregularized setup.
    custom_lr = LogisticRegression(lambda_=1e9)
    custom_lr.fit(X.T, y, iterations=15) # Give it enough iterations to converge

    # Fit Sklearn Model
    # penalty=None turns off regularization to match our math
    sk_lr = SklearnLogisticRegression(penalty=None, solver='lbfgs')
    sk_lr.fit(X, y)

    # --- Compare Weights and Bias ---
    custom_weights = custom_lr.get_weights()
    custom_bias = custom_lr.get_bias()
    
    sk_weights = sk_lr.coef_[0]
    sk_bias = sk_lr.intercept_[0]

    print("\nWeights Match:")
    print(np.allclose(custom_weights, sk_weights, atol=1e-4))
    
    print("Bias Match:")
    print(np.allclose(custom_bias, sk_bias, atol=1e-4))

    # --- Compare Log-Odds (Logits) ---
    custom_log_odds = custom_lr.get_log_odds(X.T)
    sk_log_odds = sk_lr.decision_function(X)
    print("Log-Odds / Decision Function Match:")
    print(np.allclose(custom_log_odds, sk_log_odds, atol=1e-4))

    # --- Compare Probabilities ---
    custom_lr_probs = custom_lr.predict_proba(X.T)
    # Sklearn returns [P(y=0), P(y=1)]. We only want P(y=1) to match our custom sigmoid.
    sk_lr_probs = sk_lr.predict_proba(X)[:, 1] 
    print("Probabilities Match:")
    print(np.allclose(custom_lr_probs, sk_lr_probs, atol=1e-4))

    # --- Compare Predictions ---
    custom_lr_preds = custom_lr.predict(X.T)
    sk_lr_preds = sk_lr.predict(X)
    print("Predictions Match exactly:")
    print(np.array_equal(custom_lr_preds, sk_lr_preds))
    """def get_odds(self, X):
        probs = self.get_probs(X)
        n_classes, n_samples = probs.shape
       
        self.odds = np.zeros((n_classes,n_samples))
        # FIX: Changed 'for x in...' to 'for i in...'
        for i in range(n_classes): 
            # FIX: Added 1e-15 to the denominator to prevent division by zero
            self.odds[i, :] = probs[i,:] / (1 - probs[i,:] + 1e-15) 
        return self.odds

    def get_log_odds(self, X):
        # FIX: Add a tiny epsilon to prevent np.log(0) which yields -inf
        self.log_odds = np.log(self.get_odds(X) + 1e-15)
        return self.log_odds"""