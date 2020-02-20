import face_recognition
import cv2
import numpy as np

#   EXECUTAR COM PYTHON3!!!

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# NO RASPBERRY A WEBCAM USB EH A 1
video_capture = cv2.VideoCapture(1)

#   CARREGA UMA IMAGEM DE AMOSTRA E APRENDE A RECONHECE-LA.
vitor_image = face_recognition.load_image_file("exemplo.jpg")
vitor_face_encoding = face_recognition.face_encodings(vitor_image)[0]

#   CARREGA UMA SEGUNDA IMAGEM DE AMOSTRA E APRENDE A RECONHECE-LA
biden_image = face_recognition.load_image_file("biden.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

#   CRIA ARRAYS DE CODIFICACAO DE FACES CONHECIDAS E SEUS NOMES
known_face_encodings = [
    vitor_face_encoding,
    biden_face_encoding
]
known_face_names = [
    "Vitor",
    "Joe Biden"
]

#   INICIALIZACAO DE VARIAVEIS
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

import stepper
a = stepper.andar()
while True:
    # Grab a single frame of video
    #   PEGA UM QUADRO DO VIDEO
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    #   REDIMENSIONA O QUADRO PARA 1/4 DO TAMANHO PARA ACELERAR O RECONHECIMENTO
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    #   CONVERTE A IMAGEM DE BGR PARA RGB (DE OPENCV PARA FACE_RECOGNITION)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        #    ENCONTRA TODAS AS FACES E SUAS CODIFICACOES NO QUADRO ATUAL
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            # VE SE O ROSTO COINCIDE COM OS ROSTOS CONHECIDOS
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

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        center = (right+left)/2

        # Draw a box around the face
        #cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        #cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        #font = cv2.FONT_HERSHEY_DUPLEX
        #cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        #print(center) #[Ytopo,Ycentro,Xcentro,Xdir] = Ycentro = 90 < Xcentro < 120
        
        if center < 200:
            a.dirAx(10)
        elif center > 440:
            a.esqAx(10)
        else:
            a.tras(30)
        
    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
