import random
import math
import numpy as np
import jsonpickle
import atexit
import json
import keyboard
from number_recognition import Weight,Neuron,Net

def format_data(old_version):
    lr = np.arange(10)

    desired_number=old_version[0]
    inputs=old_version[1:].tolist()

    transdformed_wished_output = (lr == desired_number).astype(np.int).tolist()

    return [transdformed_wished_output,inputs]


with open("./save_net.json") as f:
    print("Starting importing net...")
    net = jsonpickle.decode(f.read())
print("Import net complete")
#print(net)

data_path = ""
print("Starting importing train data...")
raw_train_data = np.loadtxt(data_path + "mnist_train.csv", delimiter=",", max_rows=10000)
print("Import train data complete")
train_data = [format_data(i) for i in raw_train_data]

print("Starting importing test data...")
raw_test_data = np.loadtxt(data_path + "mnist_test.csv", delimiter=",", max_rows=100)
print("Import test data complete")
test_data = [format_data(i) for i in raw_test_data]

net.Training(train_data,test_data)