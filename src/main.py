def main():
    path_to_image = '../resources/images/duckface_example_1.jpg'
    this_prediction = df_detector(path_to_image)

    print(f'Classification result for {path_to_image}: {this_prediction}')


if __name__ == "__main__":
    main()