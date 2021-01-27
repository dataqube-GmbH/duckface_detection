import detector_folder
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("p", type=str,
                    help="The path to the folder containing the images you want to classify")
args = parser.parse_args()
path = args.p
#example_path = "/home/roland/Schreibtisch/Auswertungen/face_rec/test_imgs/"
               #"/home/roland/workspace/duckface_detection/resources/images/"
def main():
    classification_results = detector_folder.df_detector.df_classification(args.p) #example_path

    for c in classification_results.keys():
        print('Duckface probability for {}: {:.1f}'.format(c, classification_results[c], ))


if __name__ == "__main__":
    main()
