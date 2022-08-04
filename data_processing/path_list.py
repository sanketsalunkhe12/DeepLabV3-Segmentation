import numpy as np

def create_path_list(file_path):
    
    with open(file_path) as f:
        path_list = f.read().split()
        path_list = np.array(path_list).reshape(int(len(path_list)/2),2)

    input_list = path_list[:,0]
    target_list = path_list[:,1]

    return input_list, target_list


def create_test_path_list(test_file_path):

    with open(test_file_path) as f:
        test_path_list = f.read().split()
        test_path_list = np.array(test_path_list)

    return test_path_list