import os


def path_test_file(filename: str):
    module = os.path.abspath(__file__)
    directory = os.path.dirname(module)
    test_data_dir = os.path.join(directory, "data")
    test_file = os.path.join(test_data_dir, filename)

    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file {test_file} does not exist.")

    return test_file