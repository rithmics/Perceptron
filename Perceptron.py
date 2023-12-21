class Perceptron():
    def __init__(self, epochs=50, rate=0.001, tolerance=0.001):
        self.epochs = epochs
        self.rate = rate
        self.tolerance = tolerance
    
    def fit(self, X, y):
        # initialize weights to zero
        self.weights_ = np.zeros(X.shape[1] + 1)
        
        net_input = self._net_input(X)
        
        for i in range(self.epochs):
            y_pred = self.predict(net_input)
            error = y - y_pred
            
            # calculate change in weights
            delta = self.rate * error.dot(net_input)
            print(delta.shape)
            
            # if change in weights is less than tol, stop training
            if delta.all() < self.tolerance:
                return self
            # else, update weights
            self.weights_ += delta
        return self
    
    def _net_input(self, X):
        bias = np.ones((X.shape[0], 1))
        z = np.c_[X, bias]
        self.z_ = z
        return z
    
    def predict(self, z):
        #if not self.z_:
            #z = self._net_input(z)
        y_pred = np.where(z.dot(self.weights_.T)>=0, 1, 0)
        return y_pred
    
    def accuracy(self, y_pred, y_true):
        score = sum(map(lambda x, y: x == y, y_true, y_pred))/len(y_true)
        return score
        
        
        
        
        