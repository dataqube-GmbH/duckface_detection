# +
import face_recognition
import cv2
import numpy as np
import talos
import detector_file

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("p", type=str,
                    help="The path to the folder containing an image of your face "
                         "(if you want the algorithm to identify you)")
parser.add_argument("n", type=str,
                    help="The Name you want to be shown")
parser.add_argument("t", type=str,
                    help="The duckface threshold, should be between 0 and 1. The higher the threshold, the more likely "
                         "a duckface alarm is issued")

args = parser.parse_args()
path = args.p
name_shown = args.n
this_threshold = args.t
restore = talos.Restore("../resources/trained_classifier/talos_duckface.zip")
#duck_img = image = cv2.imread("/home/roland/Schreibtisch/Auswertungen/face_rec/duckfacing/kisspng-mallard-duck-goose-bird-duck-5ab3b80da08ce0.1029030415217275016576.jpeg")

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
roland_image = face_recognition.load_image_file(path)
roland_face_encoding = face_recognition.face_encodings(roland_image)[0]

# Load a second sample picture and learn how to recognize it.
#biden_image = face_recognition.load_image_file("biden.jpg")
#biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    roland_face_encoding,
    #biden_face_encoding
]
known_face_names = [
    name_shown,
    #"Joe Biden"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# +
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)
            
###########add duckface detector #####################################################################################################################################
        pred = detector_file.df_detector.df_classification(rgb_small_frame,threshold=this_threshold, predict_proba=False)
        print(pred)
    #################################################################################################################################################

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
#################################################################################################################################################
        if pred == 1:
            cv2.putText(frame, "DUCKFACE DETECTED!!!", (left - 6, bottom - 42), font, 1.0, (255, 255, 255), 1)
            #frame[left:right,bottom:top,:] = duck_img[left:right,bottom:top,:]
#################################################################################################################################################
    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
