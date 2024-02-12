import numpy as np


class KNNClassifier:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    
    def __init__(self, k=1):
        self.k = k
    

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y
 

    def predict(self, X, n_loops=0):
        """
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        """
        
        if n_loops == 0:
            distances = self.compute_distances_no_loops(X)
        elif n_loops == 1:
            distances = self.compute_distances_one_loops(X)
        else:
            distances = self.compute_distances_two_loops(X)
        
        if len(np.unique(self.train_y)) == 2:
            return self.predict_labels_binary(distances)
        else:
            return self.predict_labels_multiclass(distances)


    def compute_distances_two_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        
        num_test_samples = X.shape[0]
        num_train_samples = self.train_X.shape[0]
        distances = np.zeros((num_test_samples, num_train_samples))

        for i in range(num_test_samples):
            for j in range(num_train_samples):
                distances[i, j] = np.sum(np.abs(X[i] - self.train_X[j]))

        return distances


    def compute_distances_one_loop(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """

        num_test_samples = X.shape[0]
        num_train_samples = self.train_X.shape[0]
        distances = np.zeros((num_test_samples, num_train_samples))

        for i in range(num_test_samples):
            distances[i, :] = np.sum(np.abs(X[i] - self.train_X), axis=1)

        return distances


    def compute_distances_no_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """

        num_test_samples = X.shape[0]
        num_train_samples = self.train_X.shape[0]

        X_extended = X[:, np.newaxis, :]
        X_train_extended = self.train_X[np.newaxis, :, :]

        distances = np.sum(np.abs(X_extended - X_train_extended), axis=2)

        return distances


    def predict_labels_binary(self, distances):
        """
        Returns model predictions for binary classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
           
        # inspired by https://ru.stackoverflow.com/questions/1518345/%D0%9F%D0%BE%D0%BC%D0%BE%D0%B3%D0%B8%D1%82%D0%B5-%D0%BD%D0%B0%D0%BF%D0%B8%D1%81%D0%B0%D1%82%D1%8C-%D0%BC%D0%B5%D1%82%D0%BE%D0%B4-%D0%BC%D0%BD%D0%BE%D0%B6%D0%B5%D1%81%D1%82%D0%B2%D0%B5%D0%BD%D0%BD%D0%BE%D0%B9-%D0%BA%D0%BB%D0%B0%D1%81%D1%81%D0%B8%D1%84%D0%B8%D0%BA%D0%B0%D1%86%D0%B8%D0%B8-k-nearest-neighbours-python
        """

        n_train = distances.shape[1]
        n_test = distances.shape[0]
        predictions = np.zeros(n_test)

        for i in range(n_test):
            closest_y = []
            distance = np.argsort(distances[i])[:self.k]
            closest_y = self.train_y[distance]
            predictions[i] = np.argmax(np.bincount(closest_y))

        return predictions.astype(int)


    def predict_labels_multiclass(self, distances):
        """
        Returns model predictions for multi-class classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        """

        n_train = distances.shape[0]
        n_test = distances.shape[0]
        prediction = np.zeros(n_test, int)

        for i in range(n_test):
            closest_indices = np.argsort(distances[i])[:self.k]
            closest_labels = self.train_y[closest_indices]
            label_counts = np.bincount(closest_labels)
            prediction[i] = np.argmax(label_counts)

        return prediction
