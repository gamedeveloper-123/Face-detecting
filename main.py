import cv2 #BGR

cascPath="haarcascade_frontalface_default.xml"
faceCascade=cv2.CascadeClassifier(cascPath)


img=cv2.imread("Face Detecting.jpg")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#colored=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB )
cv2.imshow("grayscale",gray)


faces=faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=3,
    minSize=(30,30),
    flags=cv2.CASCADE_SCALE_IMAGE
)

print("Found {0} faces".format(len(faces)))

print("faces=",faces)


for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),10)

cv2.imshow("faces found",img)
cv2.waitKey(0)
