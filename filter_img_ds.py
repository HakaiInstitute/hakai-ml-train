from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm


def filter_blank_images(dataset):
    imgs_dir = Path(dataset).joinpath("x")
    labels_dir = Path(dataset).joinpath("y")

    imgs = list(imgs_dir.glob("*.png"))
    print(len(list(imgs)), "Total images in dataset", dataset)

    removed = 0
    for img_path in tqdm(imgs, total=len(imgs)):
        img = Image.open(img_path)
        if img.getbbox() is None:
            # Delete label files
            labels_dir.joinpath(img_path.name).unlink()
            labels_dir.joinpath(img_path.with_suffix(".png.aux.xml").name).unlink()

            # Delete img files
            imgs_dir.joinpath(img_path.with_suffix(".png.aux.xml").name).unlink()
            img_path.unlink()

            removed += 1

    print(removed, "images removed")


def del_extra_labels(dataset):
    imgs_dir = Path(dataset).joinpath("x")
    labels_dir = Path(dataset).joinpath("y")

    imgs = list(imgs_dir.glob("*.png"))
    print(len(list(imgs)), "Total images in dataset", dataset)

    labels = list(labels_dir.glob("*.png"))
    print(len(list(labels)), "Total labels in dataset", dataset)

    img_names = [l.name for l in imgs]

    removed = 0
    for label in labels:
        if label.name not in img_names:
            removed += 1
            # Delete label files
            label.with_suffix(".png.aux.xml").unlink()
            label.unlink()

    print(removed, "labels removed")


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
