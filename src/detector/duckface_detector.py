import os
from typing import Optional

import matplotlib.image as mpimg
import matplotlib.pylab as pl
import numpy as np
import pandas as pd
import talos
from face_recognition import face_landmarks
from face_recognition import load_image_file
from matplotlib.gridspec import GridSpec


class DuckfaceDetector:
    def __init__(self, threshold: float = 0.5, predict_proba: bool = True):
        self.model = talos.Restore("../resources/trained_classifier/talos_duckface/talos_duckface.zip").model
        self.threshold = threshold
        self.predict_proba = predict_proba

    def predict_on_folder(self, folder_path: str, evalution_plot: bool = True) -> dict:
        classifications = {}
        image_list = sorted([f'{folder_path}/{file}' for file in os.listdir(folder_path)])

        for path_to_image in image_list:
            image = load_image_file(path_to_image)
            prediction = self.predict_on_image(image)
            if not prediction:
                print("could not extract mouth coordinates from", os.path.basename(path_to_image))
            classifications[path_to_image] = prediction

        if evalution_plot:
            self.create_evaluation_plot(classifications, folder_path)

        return classifications

    def predict_on_image(self, image: np.array) -> Optional[float]:
        x_input = self.get_mouth_coordinates(image)
        if x_input:
            prediction = self.model.predict(pd.DataFrame(x_input).T)[0][0]
            if not self.predict_proba:
                prediction = int(prediction >= self.threshold)
        else:
            prediction = None
        return prediction

    def get_mouth_coordinates(self, image: np.array) -> Optional[list]:
        face_landmarks_list = face_landmarks(image)
        height, width, dim = image.shape
        try:
            xtl = list(np.array([i[0] for i in face_landmarks_list[0]["top_lip"]]) / width)
            ytl = list(np.array([i[1] for i in face_landmarks_list[0]["top_lip"]]) / height)

            xbl = list(np.array([i[0] for i in face_landmarks_list[0]["bottom_lip"]]) / width)
            ybl = list(np.array([i[1] for i in face_landmarks_list[0]["bottom_lip"]]) / height)

            xch = list(np.array([i[0] for i in face_landmarks_list[0]["chin"]]) / width)
            ych = list(np.array([i[1] for i in face_landmarks_list[0]["chin"]]) / height)

            xnt = list(np.array([i[0] for i in face_landmarks_list[0]["nose_tip"]]) / width)
            ynt = list(np.array([i[1] for i in face_landmarks_list[0]["nose_tip"]]) / height)

            ratio = width / height

            # calculate mouth width, height
            mostleft_tl = min(face_landmarks_list[0]["top_lip"], key=lambda item: item[0])
            mostright_tl = max(face_landmarks_list[0]["top_lip"], key=lambda item: item[0])
            mostleft_bl = min(face_landmarks_list[0]["bottom_lip"], key=lambda item: item[0])
            mostright_bl = max(face_landmarks_list[0]["bottom_lip"], key=lambda item: item[0])
            mostleft = min((mostleft_tl, mostleft_bl))
            mostright = max((mostright_tl, mostright_bl))

            # middle of lips
            top_tl1 = np.median(face_landmarks_list[0]["top_lip"], axis=0)
            # maximal position of lips (y)
            top_tl = np.array([top_tl1[0], min(face_landmarks_list[0]["top_lip"], key=lambda item: item[1])[1]])
            # minimal position of lips (y)
            bottom_bl1 = np.median(face_landmarks_list[0]["bottom_lip"], axis=0)
            bottom_bl = np.array(
                [bottom_bl1[0], max(face_landmarks_list[0]["bottom_lip"], key=lambda item: item[1])[1]])

            # euclidian distance of left and right corner of lips
            mouth_width = (np.sqrt((mostright[0] - mostleft[0]) ** 2 + (mostright[1] - mostleft[1]) ** 2)) / width

            # euclidian distance of top and bottom point of lips
            mouth_height = (np.sqrt((top_tl[0] - bottom_bl[0]) ** 2 + (top_tl[1] - bottom_bl[1]) ** 2)) / height

            return list([mouth_width, mouth_height, ratio]) + xtl + ytl + xbl + ybl + xnt + ynt + xch + ych

        except:
            return None

    @staticmethod
    def create_evaluation_plot(classifications, folder_path: str):
        all_image_paths = list(classifications.keys())

        n_max_images = 10
        n_cols = 5
        n_rows = 2

        gs = GridSpec(n_rows, n_cols, wspace=0.4, hspace=0.4)
        fig = pl.figure(figsize=(10, 2 * n_rows))

        row = col = 0

        for image_path in all_image_paths[0:n_max_images]:
            ax = fig.add_subplot(gs[row, col])
            image = mpimg.imread(image_path)
            ax.imshow(image)
            ax.axis('off')

            probability = int(classifications[image_path] * 100)
            ax.set_title(f'{probability}% duckface', fontsize=14)

            col += 1
            if col == n_cols:
                row += 1
                col = 0

        output_folder = 'results'
        os.makedirs(output_folder, exist_ok=True)
        input_folder_base = folder_path.split('/')[-1]
        fig.savefig(f'{output_folder}/duckface_probabilities_{input_folder_base}.jpg', bbox_inches='tight')
