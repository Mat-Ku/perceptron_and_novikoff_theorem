import numpy as np


class Perceptron(object):

    def __init__(self, eta):
        self.eta = eta  # learning rate
        self.n = None   # number of iterations
        self.w = None   # weight
        self.b = None   # bias
        self.r = None   # Euclidean distance between origin of coordinate system and furthest datapoint

    @staticmethod
    def euclidean_distance(x1, x2):
        """
        Compute Euclidean distance between two points in a two-dimensional space.

        :param x1: First datapoint
        :param x2: Second datapoint

        :return: Euclidean distance as float
        """
        return np.sqrt(np.sum((x1-x2) ** 2))

    def fit(self, x_train, y_train):
        """
        Run perceptron algorithm in order to find a linear equation that separates the two different classes contained
        in 'x_train'.

        :param x_train: Training data
        :param y_train: Labels

        :return: None
        """
        self.n = 0
        self.w = np.array([0, 0])  # initialize weight as zero
        self.b = 0  # initialize bias as zero
        self.r = max([self.euclidean_distance(np.array([0, 0]), x) for x in x_train])
        correct_classifications = None

        while not correct_classifications == len(x_train):
            correct_classifications = 0
            for i in range(len(x_train)):
                # Update rule
                if y_train[i] * (np.dot(self.w, x_train[i]) + self.b) <= 0:
                    self.w = self.w + self.eta * y_train[i] * x_train[i]
                    self.b = self.b + self.eta * y_train[i] * self.r ** 2
                else:
                    correct_classifications += 1
                self.n += 1

    def get_perceptron_equation(self, x):
        """
        Retrieves the equation of the separation line fitted by perceptron algorithm.

        :param x: Domain of the perceptron equation function

        :return: Line equation
        """
        return (-self.w[0]/self.w[1])*x+(-self.b/self.w[1])

    def get_iterations(self):
        """
        Retrieve number of iterations needed by Perceptron algorithm in order to find separating line.

        :return: Number of iterations as integer.
        """
        return self.n

    def get_equal_distance_margin(self, x_train_positive, x_train_negative):
        """
        Calculates the equal distance margin between the two classes, denoted as 'gamma'. The separating function found
        by the perceptron algorithm does not necessarily have an equal distance to both classes. However, in order to
        check, whether the Novikoff theorem holds, the potential equal distance margin must be determined.
        This is done by finding a line, that has the slope of the normal of the separating line, for each training data
        instance, and determine its bias in such a way, that it runs through the respective data point. Once this normal
        is found, the point of intersection between the normal and the separating line is determined. Finally, the
        distance between each training data instance and the intersection is calculated along the respective normal. The
        shortest distance of each class is chosen. Both are added and divided by two, thereby yielding the equal
        distance margin.

        :param x_train_positive: Training instances belonging to the positive class
        :param x_train_negative: Training instances belonging to the negative class

        :return: Equal distance margin as float
        """
        # Save slope and bias of separating line
        m = -self.w[0]/self.w[1]
        b = -self.b/self.w[1]

        # Define slope of normal of separating line
        m_normal = -1/m

        # Determine biases of all normals running through training data points, separated class-wise
        b_normal_positive = [x_train_positive[i][1]-m_normal*x_train_positive[i][0] for i in range(len(x_train_positive))]
        b_normal_negative = [x_train_negative[i][1]-m_normal*x_train_negative[i][0] for i in range(len(x_train_negative))]

        # Calculate x and y values of each intersection point between each normal and the separating line
        intersections_positive_x = [(b-b_normal_positive[i])/(m_normal-m) for i in range(len(b_normal_positive))]
        intersections_positive_y = [m*intersections_positive_x[i]+b for i in range(len(intersections_positive_x))]
        intersections_positive = [np.array([intersections_positive_x[i], intersections_positive_y[i]]) for i in range(len(intersections_positive_x))]
        intersections_negative_x = [(b-b_normal_negative[i])/(m_normal-m) for i in range(len(b_normal_negative))]
        intersections_negative_y = [m*intersections_negative_x[i]+b for i in range(len(intersections_negative_x))]
        intersections_negative = [np.array([intersections_negative_x[i], intersections_negative_y[i]]) for i in range(len(intersections_negative_x))]

        # Retrieve the distance of the point of each class that is closest to the separating line with respect to the
        # separating line
        dist_positive = min([self.euclidean_distance(x_train_positive[i], intersections_positive[i]) for i in range(len(x_train_positive))])
        dist_negative = min([self.euclidean_distance(x_train_negative[i], intersections_negative[i]) for i in range(len(x_train_negative))])

        return (dist_positive+dist_negative)/2

    def check_novikoff_theorem(self, positive_instances, negative_instances):
        """
        Check whether the Novikoff theorem holds.
        The Novikoff theorem says that, as long as the data is linearly separable, the number of iterations needed by
        the perceptron algorithm in order to find a separating line, does not exceed r**2 / gamma**2.

        :param positive_instances: Training instances belonging to the positive class
        :param negative_instances: Training instances belonging to the negative class

        :return: True if Novikoff theorem holds, False otherwise
        """
        gamma = self.get_equal_distance_margin(x_train_positive=positive_instances, x_train_negative=negative_instances)

        if self.n <= (self.r**2 / gamma**2):
            return True
        else:
            return False

    def predict(self, x_test):
        """
        Uses fitted perceptron for predicting unseen data instances.

        :param x_test: Test data instances

        :return: Predicted class as integer
        """
        return [-1 if (np.dot(self.w, x_test[i]) + self.b) < 0 else 1 for i in range(len(x_test))]
