import numpy as np
import matplotlib.pyplot as plt

from perceptron_algorithm import Perceptron


# Load linearly separable data
linearly_separable_data = np.loadtxt('data/linearly_separable_data.csv', delimiter=';')
training_data = linearly_separable_data[:8, :]
test_data = linearly_separable_data[8:, :]

# Fit perceptron on training data instances and predict test data instances
perceptron = Perceptron(eta=0.01)
perceptron.fit(x_train=training_data[:, :2], y_train=training_data[:, 2])
predictions = list(zip(test_data[:, :2], perceptron.predict(x_test=test_data[:, :2])))

# Check if Novikoff theorem holds
print(perceptron.check_novikoff_theorem(positive_instances=[np.array([x[0], x[1]]) for x in training_data if x[2] == 1],
                                        negative_instances=[np.array([x[0], x[1]]) for x in training_data if x[2] == -1]))

# Plot fitted perceptron and prediction
plt.plot(np.arange(0, 5, 1), perceptron.get_perceptron_equation(x=np.arange(0, 5, 1)), color='black')
plt.scatter(x=[x[0] for x in training_data if x[2] == -1],
            y=[x[1] for x in training_data if x[2] == -1],
            c='blue', marker="o", label='negative class training instances')
plt.scatter(x=[x[0] for x in training_data if x[2] == 1],
            y=[x[1] for x in training_data if x[2] == 1],
            c='blue', marker="P", label='positive class training instances')
for i in range(len(predictions)):
    plt.scatter(x=predictions[i][0][0],
                y=predictions[i][0][1],
                c='red', marker="o" if predictions[i][1] == -1 else "P",
                label='test instance predicted as negative' if predictions[i][1] == -1
                else 'test instance predicted as positive')
plt.title('Fitted Separating Line after {} Iterations'.format(perceptron.get_iterations()))
plt.legend()
plt.show()
