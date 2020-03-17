from utils.data_prep import  filter_blank_images, del_extra_labels


def main():
    datasets = [
        "./data/datasets/Calvert_WestBeach_2016",
        "./data/datasets/Calvert_ChokedNorthBeach_2016",
        "./data/datasets/Calvert_2015",
        "./data/datasets/Calvert_2012",
    ]

    # for dataset in datasets:
    #     filter_blank_images(dataset)

    for dataset in datasets:
        del_extra_labels(dataset)


if __name__ == '__main__':
    main()
