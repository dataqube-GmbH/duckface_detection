import cv2
import face_recognition
from detector.duckface_detector import DuckfaceDetector


def main():
    duckface_classifier = DuckfaceDetector(predict_proba=False)
    video_capture = cv2.VideoCapture(0)

    process_this_frame = True

    while True:
        face_locations = []
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
            prediction = duckface_classifier.predict_on_image(rgb_small_frame)

        process_this_frame = not process_this_frame

        # Display the results
        for top, right, bottom, left in face_locations:
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            font = cv2.FONT_HERSHEY_DUPLEX
            if prediction == 1:
                cv2.putText(frame, "DUCKFACE DETECTED!!!", (left - 6, bottom - 42), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
