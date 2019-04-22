import face_alignment
import cv2

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,enable_cuda=False,flip_input=False)

input = cv2.imread('C:/Users/IVCL-34/Desktop/testImage/1.jpg')
input = cv2.resize(input,None,fx=0.5,fy=0.5, interpolation = cv2.INTER_AREA)

preds = fa.get_landmarks(input)
pl = tuple(preds)

output = input

for i in range(0,67):
    output = cv2.circle(output,(pl[0][i][0],pl[0][i][1]),3,(0,0,255),-1)
    
cv2.imshow('output',output)
cv2.waitKey(0)
cv2.destroyAllWindows()