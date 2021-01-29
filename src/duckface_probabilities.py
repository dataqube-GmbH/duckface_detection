import argparse

from detector.duckface_detector import DuckfaceDetector


def main():
    arguments = parse_arguments()

    classifications = DuckfaceDetector().predict_on_folder(arguments.path_to_image_folder)

    for c in classifications.keys():
        print('Duckface probability for {}: {:.1f}'.format(c, classifications[c]))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_image_folder", type=str,
                        help="The path to the folder containing the images you want to classify")
    return parser.parse_args()


if __name__ == "__main__":
    main()
