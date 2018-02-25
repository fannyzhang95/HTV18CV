import face_recognition
import cv2
import os

def load_face_encodings (dirname, known_face_encodings, known_face_names):
    for filename in os.listdir(dirname):
        image = face_recognition.load_image_file(dirname + '/' + filename)
        name = os.path.splitext(filename)[0].title()
        known_face_encodings.append(face_recognition.face_encodings(image)[0])
        known_face_names.append(name)

def capture_face(video_capture, dirname):
    while True:
        key = cv2.waitKey(1)
        ret, frame = video_capture.read()
        cv2.imshow("Face", frame)
        if key == ord('y'): #save on pressing 'y'
            name = input("enter your name:")
            cv2.imwrite(dirname + '/' + name + ".png",frame)
            cv2.destroyAllWindows()
        elif key == ord('q'):
            cv2.destroyAllWindows()
            break

def recognize(video_capture, known_face_encodings, known_face_names):
    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

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
    
                # If a match was found in known_face_encodings, just use the first one.
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
    
                face_names.append(name)
    
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
    
        # Display the resulting image
        cv2.imshow('Video', frame)
    
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def main():
    video_capture = cv2.VideoCapture(0)
    dirname = "faces"

    capture_face(video_capture, dirname)

    known_face_encodings = []
    known_face_names = []
    load_face_encodings(dirname, known_face_encodings, known_face_names)

    recognize(video_capture, known_face_encodings, known_face_names)

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
