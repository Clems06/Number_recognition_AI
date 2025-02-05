from new_attempt import *

with open("./save_net.json") as f:
    print("Starting importing net...")
    net = jsonpickle.decode(f.read())
print("Import net complete")

image_size = 28
no_of_different_labels = 10
image_pixels = image_size * image_size

data_path = ""
raw_train_data = np.loadtxt(data_path + "../mnist_train.csv", delimiter="," , skiprows=1000, max_rows=1000)
print("Import train data complete")
train_data = [format_data(i) for i in raw_train_data]

print(nums)

raw_test_data = np.loadtxt(data_path + "../mnist_test.csv", delimiter=",", max_rows=50)
print("Import test data complete")
test_data = [format_data(i) for i in raw_test_data]

test_sample = test_data[random.randint(0, len(test_data) - 1)]
expectedoutput_test = test_sample[1]
actualoutput = net.get_output(test_sample[0])

error = sum([(expectedoutput_test[i] - actualoutput[i]) ** 2 for i in range(len(expectedoutput_test))])
print()
print("Error {0}      Expected {1} and Got {2}".format(error, expectedoutput_test,
                                                                  actualoutput))


net.train2(train_data, test_data, batch_size=30)
