import random
import math
import numpy as np
import jsonpickle
import atexit
import json
import keyboard


class Weight:
    def __init__(self):

        #Initialise randomly
        self.weight = random.uniform(-1, 1)

        #Variables that will store the sum of the wished changes for each example and then do the average later
        self.total_wished_changes = 0
        self.number_of_wished_changes = 0


class Neuron:
    def __init__(self, layer, number, num_output_weights=0):
        #Classifying variables
        self.num_output_weights = num_output_weights
        self.layer = layer
        self.number = number

        #Value of the neuron before the sigmoid squishification function
        self.z = 0
        #Value of the neuron after the signmoid function
        self.value = 0

        #Initialise the weights comming from this neuron
        self.output_weights = []
        for i in range(self.num_output_weights):
            self.output_weights.append(Weight())

        #Value of the bias
        self.bias_weight = 0

        # Variables that will store the sum of the wished changes for the bias for each example and then do the average later
        self.total_wished_bias_changes = 0
        self.number_wished_bias_changes = 0

        #Calculated value of the derivative of the neuron with respect to the cost, times the derivative of the sigmoid
        self.delta = 0


class Net:
    def __init__(self, topology, learning_rate=0.5, bias=1):
        #Number of neurons for each layer
        self.topology = topology

        #Value that will multiply the bias_weight value of each neuron, 1 by default
        self.bias = bias

        #Self.layers is where all the neurons are stored. This is just to initialise it
        self.topology.append(0)
        self.layers = []
        for layer in range(len(self.topology) - 1):
            #Create a layer, add it a number of neurons and append it to self.layers
            layer_toAdd = []
            for i in range(self.topology[layer]):
                layer_toAdd.append(Neuron(layer, i, self.topology[layer + 1]))
            self.layers.append(layer_toAdd)
        self.topology.pop()

        self.learning_rate = learning_rate

    #Function that goes throught each of the neurons of a certain layer and calculates z, then its actual value
    def modifLayer(self, layer):
        for neuron in range(len(self.layers[layer])):
            #Z= sum of all the previous layers values * the weights + the bias
            self.layers[layer][neuron].z = sum(
                [inputNeuron.value * inputNeuron.output_weights[neuron].weight for inputNeuron in
                 self.layers[layer - 1]]) + self.bias * self.layers[layer][neuron].bias_weight

            #Apply the sigmoid function to z to get the actual value
            self.layers[layer][neuron].value = self.sigmoid(self.layers[layer][neuron].z)

    #Function that initialises the first layer according to the inputs, then calculates each layer according to its previous one
    def calculateResult(self, inputs):
        #Input layer
        for i in range(len(inputs)):
            self.layers[0][i].value = inputs[i]

        #Calculate values of each layer
        for layer in range(1, len(self.topology)):
            self.modifLayer(layer)

        #return the values of the output layer
        return [neuron.value for neuron in self.layers[-1]]

    #Sigmoid function. To avoid OverflowError, when x is too high or too low it makes automatically 1 or 0
    def sigmoid(self, x):
        if x < -100:
            return 0
        if x > 100:
            return 1
        return 1 / (1 + np.exp(-x))

    # math.exp(-z) / ((1 + math.exp(-z)) ** 2) = (1 / (1 + math.exp(-z))) * (1 - 1 / (1 + math.exp(-z))) = sigmoid * (1 - sigmoid) = value * (1 - value)
    def DerivativeSigmoid(self, value):
        return value * (1 - value)

    #Funtion that takes inputs and expected outputs and calculates the wished changes for each bias and weight
    def trainOnOneExample(self, inputs, expected_outputs):

        outputs_got = self.calculateResult(inputs)

        errors = [(expected_outputs[i] - outputs_got[i]) ** 2 for i in range(len(expected_outputs))]

        #A: Derivative of a neuron value: sum of ( (all the weights its got) x (derivative of the signmoid function) x (wished output of the neuron the weight leads to) ).
        #   As the Derivative of a neuron value is always multilied by the derivative of the sigmoid, I call this product delta and use that
        #B: Derivative of a weight: (value of the neuron it comes from) x (the derivative of the sigmoid function) x (the wished output of the neuron the weight leads to)
        #C: Derivative of the bias: (derivative of the sigmoid function) x (the wished change of that neuron)
        for idx in range(1,len(self.layers)+1):
            layer_idx = -idx
            for neuron_idx in range(len(self.layers[layer_idx])):
                neuron = self.layers[layer_idx][neuron_idx]
                #A
                if layer_idx != -1:
                    neuron_derVsCost = sum([neuron.output_weights[i].weight * self.layers[layer_idx+1][i].delta for i in range(self.topology[layer_idx+1])])
                else:
                    neuron_derVsCost = 2 * (expected_outputs[neuron_idx] - neuron.value)

                neuron.delta = neuron_derVsCost * self.DerivativeSigmoid(neuron.value)
                #C
                neuron.total_wished_bias_changes += neuron.delta
                neuron.number_wished_bias_changes += 1

                #Go throught each weight
                for weight in range(len(neuron.output_weights)):
                    this_weight = neuron.output_weights[weight]
                    #B
                    this_weight.total_wished_changes += neuron.value * self.layers[layer_idx+1][weight].delta
                    this_weight.number_of_wished_changes += 1


    def CorrectFromWished(self):
        for layer in self.layers[1:]:
            for neuron in layer:
                neuron.bias_weight += (neuron.total_wished_bias_changes / neuron.number_wished_bias_changes) * self.learning_rate
                neuron.total_wished_bias_changes = 0
                neuron.number_wished_bias_changes = 0

                for weight in neuron.output_weights:
                    weight.weight += (weight.total_wished_changes / weight.number_of_wished_changes) * self.learning_rate
                    weight.total_wished_changes = 0
                    weight.number_of_wished_changes = 0



    #Function that takes the training data and test data, trains the NN, modifies the values and test to see the improvement
    def Training(self, training_data, test_data):
        keyboard.on_press_key("g", self.save_data)
        #Divide in mini batches for stochastic gradeint descend
        random.shuffle(training_data)
        batches = np.array_split(training_data, round(len(training_data)**0.5))

        #lr = np.arange(10)

        for epoch in range(1000):
            for batch in batches:
                for sample in batch:
                    print("|",end="")

                    #Trains an example
                    expectedoutput = sample[0]
                    self.trainOnOneExample(sample[1], expectedoutput)

                self.CorrectFromWished()
                print()

            #Randomly select a test sample and print its result
            test_sample = test_data[random.randint(0, len(test_data) - 1)]
            expectedoutput_test = test_sample[0]
            actualoutput = self.calculateResult(test_sample[1])
            error = sum([(expectedoutput_test[i] - actualoutput[i])**2 for i in range(len(expectedoutput_test))])
            print("Epoch {0}: Error {1}      Expected {2} and Got {3}".format(epoch,error,expectedoutput_test,actualoutput))

    def save_data(self,evt):
        print("Initialising saving...")
        json_net = jsonpickle.encode(self)
        open('./save_net.json', 'w').close()
        with open("./save_net.json", 'w') as f:
            f.write(json_net)
        print('Net succesfully saved')



def format_data(old_version):
    lr = np.arange(10)

    desired_number=old_version[0]
    inputs=old_version[1:].tolist()

    transdformed_wished_output = (lr == desired_number).astype(np.int).tolist()

    return [transdformed_wished_output,inputs]

def main():
    image_size = 28
    no_of_different_labels = 10
    image_pixels = image_size * image_size

    data_path = ""
    raw_train_data = np.loadtxt(data_path + "mnist_train.csv", delimiter="," , max_rows=500)
    print("Import train data complete")
    train_data = [format_data(i) for i in raw_train_data]

    raw_test_data = np.loadtxt(data_path + "mnist_test.csv", delimiter=",", max_rows=50)
    print("Import test data complete")
    test_data = [format_data(i) for i in raw_test_data]

    net = Net([image_pixels, 100, 10])
    net.Training(train_data,test_data)

#Avoid creating new net if it is imported
if __file__.split("/")[-1] == "number_recognition.py":
    print("Starting new net...")
    main()