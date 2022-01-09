import cv2,numpy,os

haar_file = "haarcascade_frontalface_alt2.xml"
datasets = 'dataset'
print('Training...')
(images, labels, names, id) = ([], [], {}, 0)

for (root, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
        id +=1

(images, labels) = [numpy.array(lis) for lis in [images, labels]]
print(images, labels)
(width, height) = (130, 100)
model = cv2.face.LBPHFaceRecognizer_create()
#model =  cv2.face.FisherFaceRecognizer_create()

model.train(images, labels)

face_cascade = cv2.CascadeClassifier(haar_file)
cam = cv2.VideoCapture(0)
cnt = 0
while True:
    _, img = cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        face = gray[y:y+h,x:x+w]
        face_resize = cv2.resize(face, (width,height))

        pred = model.predict(face_resize)
        if pred[1] < 800:
            cv2.putText(img,"%s - %.0f" % (names[pred[0]],pred[1]),(x-10,y-10),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255))
            print(names[pred[0]])
            cnt = 0
        else:
            cnt+=1
            cv2.putText(img,"unknown",(x-10,y-10),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255))
            if (cnt)>100:
                print("Unknown Person")
                cv2.imwrite("Unknown.jpg",img)
                cnt=0
    cv2.imshow("Face Recognition",img)
    key = cv2.waitKey(1)
    if key == 27:
        break
cam.release()
cv2.destroyAllWindows()
