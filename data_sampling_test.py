from src.dataset import PedalDataset
from src.utils import load_data, split_data

def check_data_sampling(dataset, label_bin_edges):
    total_label_dist = {}
    for i in range(len(label_bin_edges) - 1):
        total_label_dist[i] = 0
    for data in dataset:
        feature, _, label, _, _, room, midi_id, factor = data
        # print("Features shape:", feature.shape)
        # print("Labels shape:", label.shape)
        # print("Metadata(Room):", metadata)
        # print("")
        sample_label_dist = {}
        for i in range(len(label_bin_edges) - 1):
            sample_label_dist[i] = 0
        for l in label:
            l = int(l.item())
            total_label_dist[l] += 1
            sample_label_dist[l] += 1
        # print("Sample label distribution:", sample_label_dist)
    print("Total label distribution:", total_label_dist)


def main():

    data_version = "_4096_NormPerFeat"

    # Feature dimension
    max_frame = 100
    num_classes = 128
    pedal_factor = [1.0]
    room_acoustics = [1.0]

    label_bin_edges = []
    if num_classes == 3:
        label_bin_edges = [0, 11, 95, 128]
    elif num_classes == 4:
        label_bin_edges = [0, 11, 60, 95, 128]
    elif num_classes == 2:
        label_bin_edges = [0, 11, 128]
    elif num_classes == 128:
        label_bin_edges = range(129)

    # Data path
    data_path = f"data/processed_data{data_version}.npz"

    # Load data
    features, labels, metadata = load_data(data_path, label_bin_edges, pedal_factor, room_acoustics)
    train_features, val_features, test_features, train_labels, val_labels, test_labels, train_metadata, val_metadata, test_metadata = split_data(
        features, labels, metadata, val_size=0.15, test_size=0.15, random_state=100
    )

    # Dataset and DataLoader
    train_dataset = PedalDataset(
        features=train_features,
        labels=train_labels,
        metadata=train_metadata,
        num_samples_per_clip=100,
        max_frame=max_frame,
        label_ratio=1.0,
        label_bin_edges=label_bin_edges,
        overlap_ratio=0.25,
    )
    val_dataset = PedalDataset(
        features=val_features,
        labels=val_labels,
        metadata=val_metadata,
        num_samples_per_clip=100,
        max_frame=max_frame,
        label_ratio=1.0,
        label_bin_edges=label_bin_edges,
        overlap_ratio=0.25,
    )
    test_dataset = PedalDataset(
        features=test_features,
        labels=test_labels,
        metadata=test_metadata,
        num_samples_per_clip=100,
        max_frame=max_frame,
        label_ratio=1.0,
        label_bin_edges=label_bin_edges,
        overlap_ratio=0.25,
    )
    print("Train dataset size:", len(train_dataset))
    print("Val dataset size:", len(val_dataset))
    print("Test dataset size:", len(test_dataset))

    # Check data sampling
    check_data_sampling(train_dataset, label_bin_edges)
    check_data_sampling(val_dataset, label_bin_edges)
    check_data_sampling(test_dataset, label_bin_edges)


if __name__ == "__main__":
    main()
