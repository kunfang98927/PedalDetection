import numpy as np


def main():

    # Data path
    data_path = "data/processed_data.npz"

    # Load data
    data = np.load(data_path, allow_pickle=True)
    features = data["features"]
    labels = data["labels"]
    metadata = data["metadata"]

    print("Features shape:", features.shape)
    print("Labels shape:", labels.shape)
    print("Metadata shape:", metadata.shape)

    # Count the number of samples for each class
    class_count = np.zeros(128)
    for label in labels:
        unique, counts = np.unique(label, return_counts=True)
        for i, c in zip(unique, counts):
            i = int(i)
            class_count[i] += c

    # bin_0: <= 25
    # bin_1: 26 - 100
    # bin_2: 101 - 127
    bin_count = np.zeros(3)
    for i, c in enumerate(class_count):
        if i <= 25:
            bin_count[0] += c
        elif i <= 100:
            bin_count[1] += c
        else:
            bin_count[2] += c
    print("Bin count:", bin_count)


if __name__ == "__main__":
    main()
