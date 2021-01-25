import cv2
import numpy as np
import pandas as pd
import talos
from face_recognition import load_image_file
from face_recognition import face_landmarks

img_path = '../../resources/images/duckface_example_1.jpg'
class df_detector:
    #def __init__(self, img_path):
        #self.img_path = img_path

    def df_classification(path_to_image, threshold=0.5, predict_proba=False):

        restore = talos.Restore("../../resources/trained_classifier/talos_duckface.zip")
        image = load_image_file(img_path)
        face_landmarks_list = face_landmarks(image)
        height, width, dim = image.shape

        xtl = list(np.array([i[0] for i in face_landmarks_list[0]["top_lip"]]) / width) #Normalisierung
        ytl = list(np.array([i[1] for i in face_landmarks_list[0]["top_lip"]]) /height)

        xbl = list(np.array([i[0] for i in face_landmarks_list[0]["bottom_lip"]]) /width)
        ybl = list(np.array([i[1] for i in face_landmarks_list[0]["bottom_lip"]]) /height)

        xch = list(np.array([i[0] for i in face_landmarks_list[0]["chin"]]) /width)
        ych = list(np.array([i[1] for i in face_landmarks_list[0]["chin"]]) /height)

        xnt = list(np.array([i[0] for i in face_landmarks_list[0]["nose_tip"]]) /width)
        ynt = list(np.array([i[1] for i in face_landmarks_list[0]["nose_tip"]]) /height)

        ratio = width/height

        #calculate mouth width, height
        mostleft_tl = min(face_landmarks_list[0]["top_lip"],key=lambda item:item[0])
        mostright_tl = max(face_landmarks_list[0]["top_lip"],key=lambda item:item[0])
        mostleft_bl = min(face_landmarks_list[0]["bottom_lip"],key=lambda item:item[0])
        mostright_bl = max(face_landmarks_list[0]["bottom_lip"],key=lambda item:item[0])
        mostleft = min((mostleft_tl, mostleft_bl))
        mostright = max((mostright_tl,mostright_bl))

        #Mitte der Lippe
        top_tl1 = np.median(face_landmarks_list[0]["top_lip"],axis=0)
        #maximale Position der Lippe (y)
        top_tl = np.array([top_tl1[0],min(face_landmarks_list[0]["top_lip"],key=lambda item:item[1])[1]])
        bottom_tl = np.array([top_tl1[0],max(face_landmarks_list[0]["top_lip"],key=lambda item:item[1])[1]])
        #Minimale Position der Lippe (y)
        bottom_bl1 = np.median(face_landmarks_list[0]["bottom_lip"],axis=0)
        bottom_bl = np.array([bottom_bl1[0],max(face_landmarks_list[0]["bottom_lip"],key=lambda item:item[1])[1]])

        #Distanz zwischen linker und rechter Ecke der Lippe #euklidische distanz:
        mouth_width = (np.sqrt((mostright[0] - mostleft[0])**2 + (mostright[1] - mostleft[1])**2)) / width

        #Distanz zwischen oberestem und unterstem Punkt der Lippe
        mouth_height = (np.sqrt((top_tl[0] - bottom_bl[0])**2 + (top_tl[1] - bottom_bl[1])**2)) / height

        x_input = list([mouth_width, mouth_height,ratio]) + xtl + ytl + xbl + ybl + xnt + ynt + xch + ych

        if predict_proba:
            print("predict probability")
            pred = restore.model.predict(pd.DataFrame(x_input).T)[0][0]
        else:
            print("predicting boolean...")
            pred = restore.model.predict(pd.DataFrame(x_input).T)[0][0]
            if pred > threshold:
                pred = 1
            elif pred < threshold:
                pred = 0
        return pred