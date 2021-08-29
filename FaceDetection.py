import cv2

face_cascade = cv2.CascadeClassifier('C:/Users/Dell/Desktop/Computer Vision/Module_1_Face_Recognition/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:/Users/Dell/Desktop/Computer Vision/Module_1_Face_Recognition/haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier('C:/Users/Dell/Desktop/Computer Vision/Module_1_Face_Recognition/Nariz.xml')
mouth_cascade = cv2.CascadeClassifier('C:/Users/Dell/Desktop/Computer Vision/Module_1_Face_Recognition/Mouth.xml')
def detect(gray,frame):
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces :
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_frame = frame[y:y+h,x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray,1.1,5)
        nose = nose_cascade.detectMultiScale(roi_gray,1.1,5)
        mouth = mouth_cascade.detectMultiScale(roi_gray,1.1,5)
        for(x1,y1,w1,h1) in eyes:
            cv2.rectangle(roi_frame,(x1,y1),(x1+w1,y1+h1),(0,255,0),2)
        for (x2, y2, w2, h2) in nose:
            cv2.rectangle(roi_frame, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 255), 2)
        for (x3, y3, w3, h3) in mouth:
            cv2.rectangle(roi_frame, (x3, y3), (x3 + w3, y3 + h3), (0, 255, 255), 2)
    return(frame)

video_capture = cv2.VideoCapture(0)

while(1):
    _,frame = video_capture.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    canvas = detect(gray,frame)
    cv2.imshow('Video',canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()