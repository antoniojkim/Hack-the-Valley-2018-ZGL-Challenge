from numpy import loadtxt as np_loadtxt, array as np_array

def getTrain():
    results = np_loadtxt(open("train.csv", "rb"), delimiter=",", skiprows=1, usecols=(1, 2, 3, 4))
    x_train = []
    y_train = []
    for result in results:
        x_train.append(result[0:3])
        y_train.append(result[3])
    return np_array(x_train), np_array(y_train)

def getTest():
    results = np_loadtxt(open("predict.csv", "rb"), delimiter=",", skiprows=1, usecols=(1, 2, 3, 4))
    x_test = []
    y_test = []
    for result in results:
        x_test.append(result[0:3])
        y_test.append(result[3])

        # x_test.append([result[0]-result[2], result[1]-result[2], 0])
        # y_test.append(result[3]-result[2])
    return np_array(x_test), np_array(y_test)

if __name__ == "__main__":
    # (756, 3)
    # (756,)
    # (244,)
    # (0,)
    # (756, 3)
    # (756,)
    # (122, 3)
    # (122,)
    x_train, y_train = getTrain()
    # print(x_train)
    # print(y_train)
    x_test, y_test = getTest()
    # print(x_test)
    # print(y_test)