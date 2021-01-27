import numpy as np
import pandas as pd
import talos
from face_recognition import face_landmarks

#exple_path = '../resources/images/duckface_example_1.jpg'
#example_path = glob.glob("/home/roland/workspace/duckface_detection/resources/images/*")[0]
#image = load_image_file(img_path)
class df_detector:

    def df_classification(image, threshold=0.5):
        restore = talos.Restore("../resources/trained_classifier/talos_duckface/talos_duckface.zip")
        face_landmarks_list = face_landmarks(image)
        height, width, dim = image.shape
        pred = 0
        #try:
        xtl = list(np.array([i[0] for i in face_landmarks_list[0]["top_lip"]]) / width)  # Normalisierung
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

        # Mitte der Lippe
        top_tl1 = np.median(face_landmarks_list[0]["top_lip"], axis=0)
        # maximale Position der Lippe (y)
        top_tl = np.array([top_tl1[0], min(face_landmarks_list[0]["top_lip"], key=lambda item: item[1])[1]])
        # Minimale Position der Lippe (y)
        bottom_bl1 = np.median(face_landmarks_list[0]["bottom_lip"], axis=0)
        bottom_bl = np.array([bottom_bl1[0], max(face_landmarks_list[0]["bottom_lip"], key=lambda item: item[1])[1]])

        # Distanz zwischen linker und rechter Ecke der Lippe #euklidische distanz:
        mouth_width = (np.sqrt((mostright[0] - mostleft[0]) ** 2 + (mostright[1] - mostleft[1]) ** 2)) / width

        # Distanz zwischen oberestem und unterstem Punkt der Lippe
        mouth_height = (np.sqrt((top_tl[0] - bottom_bl[0]) ** 2 + (top_tl[1] - bottom_bl[1]) ** 2)) / height

        x_input = list([mouth_width, mouth_height, ratio]) + xtl + ytl + xbl + ybl + xnt + ynt + xch + ych

        pred = restore.model.predict(pd.DataFrame(x_input).T, verbose=0)[0][0]
        if pred > float(threshold):
            pred = 1
        elif pred < float(threshold):
            pred = 0
        #except:
         #   print("could not extract face")
        return pred
