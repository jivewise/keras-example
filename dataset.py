"""Class that holds the data for training and testing."""

from numpy import squeeze, unique
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight

class DataSet():
    """Training and test data class that can help you scale features and unscale predictions"""

    def __init__(self, X_train, X_test, y_train, y_test, y_scaler=None):
        """Initialize our dataset.

        Args:
            X_train (Pandas DataFrame): Features for training
            X_test (Pandas DataFrame): Features for testing
            y_train (Pandas Series): Output for training
            y_test (Pandas Series): Output for testing
            y_scaler (MinMaxScaler): Scaler for use when unscaling
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_scaler = y_scaler

    def scale(self, predict_value):
        """Scale the data in our dataset to work with neural networks

        Args:
            predict_value (Boolean): Set to true if y is a value rather than a class
        """
        print("Scaling data***********************")
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler = scaler.fit(self.X_train)
        self.X_train = scaler.transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

        y_scaler = MinMaxScaler(feature_range=(0, 1))
        # fit_transform and transform expect matricies, so we reshape it,
        # transform, and the turn it back into a vector using squeeze
        y_train_scaled = squeeze(y_scaler.fit_transform(
            self.y_train.values.reshape(-1, 1))) if predict_value else None
        y_test_scaled = squeeze(y_scaler.transform(
            self.y_test.values.reshape(-1, 1))) if predict_value else None

        if (predict_value):
            self.y_train = y_train_scaled
            self.y_test = y_test_scaled
            self.y_scaler = y_scaler

    def unscale_y(self, y):
        """Inverse scale the series passed in so we can compare to the original predictions

        Args:
            y (Pandas Series): Series to scale back
        """
        if (not self.y_scaler):
            return y

        y_unscaled = self.y_scaler.inverse_transform(y.reshape(-1, 1))
        return y_unscaled

    def group_timesteps(self, timesteps):
        """Group training and test data by timesteps, helpful for LSTM networks

        Args:
            timesteps (int): Number of timesteps to group by
        """
        # return if it's already been reshaped
        if (len(self.X_test.shape) > 2):
            return
        self.X_test = self.X_test.reshape(self.X_test.shape[0], timesteps, self.X_test.shape[1])
        self.X_train = self.X_train.reshape(self.X_train.shape[0], timesteps, self.X_train.shape[1])

    def class_weights(self):
        """Get the class weights for this particular class

        Returns:
            class_weight_vect : ndarray, shape (n_classes,)
        """
        return class_weight.compute_class_weight('balanced',
                                                          unique(self.y_train),
                                                          self.y_train)
