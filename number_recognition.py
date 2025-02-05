import numpy as np
import random
import time
import keyboard
import jsonpickle

start_time = time.time()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def der_sigmoid(x):
    ex = np.exp(-x)
    return ex / ((1 + ex) ** 2)

def leaky_ReLu(x):
    return x * 0.01 if x < 0 else x


def der_leaky_ReLu(x):
    return 0.01 if x < 0 else 1


class Net:
    def __init__(self, topology, learning_rate=0.1):
        self.topology = topology
        self.learning_rate = learning_rate
        plage_random = 0.001
        self.weights = [np.random.rand(topology[i + 1], topology[i])*plage_random*2-plage_random for i in range(len(topology) - 1)]
        self.biases = [None] + [np.random.rand(topology[i + 1])*plage_random*2-plage_random for i in range(len(topology) - 1)]

        """self.activation = np.vectorize(leaky_ReLu)
        self.der_activation = np.vectorize(der_leaky_ReLu)"""
        self.activation = np.vectorize(sigmoid)
        self.der_activation = np.vectorize(der_sigmoid)

        self.weight_derivative = [np.zeros((topology[i + 1], topology[i])) for i in range(len(topology) - 1)]
        self.bias_derivative = [None] + [np.zeros(topology[i + 1]) for i in range(len(topology) - 1)]

    def get_output(self, input: np.ndarray):
        if len(input) != self.topology[0]:
            raise ValueError("Expected input size {0} but got {1}".format(self.topology[0], len(input)))

        values = input
        for i in range(len(self.weights)):
            values = self.activation(np.matmul(self.weights[i], values) + self.biases[i + 1])

        return values

    def with_backpropagation(self, input: np.ndarray, expected_output: np.ndarray):
        if len(input) != self.topology[0]:
            raise ValueError("Expected input size {0} but got {1}".format(self.topology[0], len(input)))
        if len(expected_output) != self.topology[-1]:
            raise ValueError("Expected output size {0} but got {1}".format(self.topology[1], len(expected_output)))

        layer_values = [input]
        z_values = [0]
        values = input
        for i in range(len(self.weights)):
            z = np.clip(np.matmul(self.weights[i], values) + self.biases[i + 1], -300, 300)
            values = self.activation(z)
            z_values.append(z)
            layer_values.append(values)

        values_vs_cost = [None] * (len(self.topology) - 1) + [2 * (layer_values[-1] - expected_output)]
        for i in range(len(self.topology) - 1, 0, -1):
            der_z = self.der_activation(z_values[i])

            self.bias_derivative[i] += der_z * values_vs_cost[i]
            values_vs_cost[i-1] = np.array([np.sum(self.weights[i-1][:, j]* der_z *values_vs_cost[i]) for j in range(self.topology[i-1])])
            self.weight_derivative[i - 1] += np.array([[layer_values[i - 1][k] * der_z[j] * values_vs_cost[i][j] for k in range(self.topology[i - 1])] for j in range(self.topology[i])])

    def reset_derivatives(self):
        self.weight_derivative = [np.zeros((self.topology[i + 1], self.topology[i])) for i in
                                  range(len(self.topology) - 1)]
        self.bias_derivative = [None] + [np.zeros(self.topology[i + 1]) for i in range(len(self.topology) - 1)]

    def train(self, training_data, epochs=500, batch_size=20):
        for epoch in range(epochs):
            print("epoch=", epoch)
            for _ in range(batch_size):
                input, output = training_data[random.randint(0, len(training_data) - 1)]
                self.with_backpropagation(np.array(input), np.array(output))

            for a in range(len(self.topology)):
                if a != len(self.topology) - 1:
                    self.weights[a] -= self.weight_derivative[a] / batch_size * self.learning_rate
                if a != 0:
                    self.biases[a] -= self.bias_derivative[a] / batch_size * self.learning_rate

            self.reset_derivatives()

    def train2(self, training_data, test_data, epochs=500, batch_size=20):
        #keyboard.on_press_key("g", self.save_data)

        for epoch in range(epochs):
            print("epoch=",epoch)
            permutation = np.random.permutation(training_data)

            for i in range(len(training_data)):
                print("|", end="")

                input, output = permutation[i]

                self.with_backpropagation(np.array(input), np.array(output))

                if (i!=0 and i%batch_size == 0) or i == len(training_data)-1:
                    print()
                    for a in range(len(self.topology)):
                        if a != len(self.topology) - 1:
                            self.weights[a] -= self.weight_derivative[a] / batch_size * self.learning_rate
                        if a != 0:
                            self.biases[a] -= self.bias_derivative[a] / batch_size * self.learning_rate

                    self.reset_derivatives()

            test_sample = test_data[random.randint(0, len(test_data) - 1)]
            expectedoutput_test = test_sample[1]
            actualoutput = self.get_output(test_sample[0])

            error = sum([(expectedoutput_test[i] - actualoutput[i]) ** 2 for i in range(len(expectedoutput_test))])
            print()
            print("Epoch {0}: Error {1}      Expected {2} and Got {3}".format(epoch, error, expectedoutput_test,
                                                                              actualoutput))

            self.save_data()

        self.save_data()

    def save_data(self,evt=None):
        print("Initialising saving...")
        json_net = jsonpickle.encode(self)
        open('./save_net.json', 'w').close()
        with open("./save_net.json", 'w') as f:
            f.write(json_net)
        print('Net succesfully saved')


nums = [0]*10
def format_data(old_version):
    lr = np.arange(10)

    desired_number=old_version[0]
    nums[int(desired_number)] += 1
    inputs=old_version[1:].tolist()

    return [inputs, (lr == desired_number).astype(np.int).tolist()]

if __name__ == "__main__":
    image_size = 28
    no_of_different_labels = 10
    image_pixels = image_size * image_size

    data_path = ""
    raw_train_data = np.loadtxt(data_path + "../mnist_train.csv", delimiter="," , max_rows=5000)
    print("Import train data complete")
    train_data = [format_data(i) for i in raw_train_data]

    print(nums)

    raw_test_data = np.loadtxt(data_path + "../mnist_test.csv", delimiter=",", max_rows=50)
    print("Import test data complete")
    test_data = [format_data(i) for i in raw_test_data]

    net = Net([image_pixels, 300, 100, 10])
    net.train2(train_data, test_data, batch_size=30)


    """test = Net([3, 2, 2])
    
    examples = [[[0, 0, 1], [0, 1]], [[1, 0, 1], [1, 1]], [[0, 1, 1], [1, 1]], [[0, 0, 0], [0, 0]], [[1, 0, 0], [1, 0]], [[1, 1, 0], [0, 0]]]
    
    
    
    test.train(examples, 2000)

    #test.train(examples)
    
    print("After")
    print(test.get_output(np.array([1, 1, 1])))
    print(test.get_output(np.array([1, 1, 0])))
    print(test.get_output(np.array([0, 1, 0])))
    print(test.get_output(np.array([0, 1, 1])))
    print(test.get_output(np.array([1, 0, 0])))"""


    print("Took {} seconds".format(time.time()-start_time))
