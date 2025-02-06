import numpy as np


def main():

    # Data path
    data_path = "data/processed_data1.npz"

    # Load data
    data = np.load(data_path, allow_pickle=True)
    features = data["features"]
    labels = data["labels"]
    metadata = data["metadata"]

    print("Features shape:", features.shape)
    print("Labels shape:", labels.shape, labels[0])
    print("Metadata shape:", metadata.shape, metadata[0])

    # Count the number of samples for each class
    class_count_0 = np.zeros(128)
    class_count_1 = np.zeros(128)
    class_count_2 = np.zeros(128)
    for i, label in enumerate(labels):
        unique, counts = np.unique(label, return_counts=True)
        print(i, metadata[i])
        for j, c in zip(unique, counts):
            j = int(j)
            if metadata[i][0] == 1:
                class_count_0[j] += c
            elif metadata[i][0] == 2:
                class_count_1[j] += c
            else:
                class_count_2[j] += c

    # plot class_count as histogram
    import matplotlib.pyplot as plt
    # plot three class_count in three plots
    fig, axs = plt.subplots(1, 3, figsize=(12, 3))
    titles = ["Dry room no reverb", "Clean studio moderate reverb", "Large concert hall max reverb"]
    for i, class_count in enumerate([class_count_0, class_count_1, class_count_2]):
        axs[i].bar(range(128), class_count)
        axs[i].set_title("Class count in " + titles[i])
        axs[i].set_xlabel("Class")
        axs[i].set_ylabel("Count")
    plt.tight_layout()

    # save figure
    plt.savefig("class_count.png")

    # bin_0: <= 10
    # bin_1: 10 - 100
    # bin_2: 101 - 127
    bin_count = np.zeros(3)
    for i, c in enumerate(class_count):
        if i <= 10:
            bin_count[0] += c
        elif i <= 100:
            bin_count[1] += c
        else:
            bin_count[2] += c
    print("Bin count:", bin_count)


if __name__ == "__main__":
    main()
