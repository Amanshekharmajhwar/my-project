import cv2
# Load model files
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

# ageProto = "age_deploy.prototxt"
# ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

# Load the networks
faceNet = cv2.dnn.readNet(faceModel, faceProto)
# ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Model mean values and other settings
# MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
MODEL_MEAN_VALUES =(78.4263377603,)

# ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)',
#            '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Initialize webcam
cap = cv2.VideoCapture(0)

def detect_face(frame):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detections = faceNet.forward()
    h, w = frame.shape[:2]
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            bboxes.append([x1, y1, x2, y2])
    return bboxes

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detect_face(frame)

    for box in faces:
        x1, y1, x2, y2 = box
        face = frame[max(0, y1-10):min(y2+10, frame.shape[0]-1),
                     max(0, x1-10):min(x2+10, frame.shape[1]-1)]

        # Blob for age & gender models
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                     MODEL_MEAN_VALUES, swapRB=False)

        # Predict gender
        genderNet.setInput(blob)
        genderPred = genderNet.forward()
        gender = genderList[genderPred[0].argmax()]

        # Predict age
        # ageNet.setInput(blob) 
        # agePred = ageNet.forward()
        # age = ageList[agePred[0].argmax()]

        # label = f"{gender}, {age}"
        label =f"{gender}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow(" Gender Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
