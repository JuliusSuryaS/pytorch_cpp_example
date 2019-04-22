import os, sys
import cv2
import face_alignment
import numpy as np
from skimage import io
import warnings
import matplotlib.pyplot as plt

if not sys.warnoptions:
    warnings.simplefilter("ignore")

def draw_landmark(img_path, landmark):
    img = cv2.imread(img_path)
    for i in range(68):
       cv2.circle(img, (landmark[i,0], landmark[i,1]), 1, (0,0,255)) 
    return im

def prediction(frontal, profile, landmark_path, landmark_profile_path):
    print(frontal, profile)
    print(landmark_path, landmark_profile_path)
    # Run the 3D face alignment on a test image, without CUDA.
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, enable_cuda=False, flip_input=False,use_cnn_face_detector=True)
    frontal_img = io.imread(frontal)
    profile_img = io.imread(profile)
    frontal_landmark = fa.get_landmarks(frontal_img)[-1]
    profile_landmark = fa.get_landmarks(profile_img)[-1]
    #print(preds[0,0])
    #print(preds[0,1])
    #print(preds[67,0])
    #write to txt
 
    print(landmark_path)
    file=open(landmark_path,'w')
    for i in range(0,68):
        file.write("%d " %frontal_landmark[i,0])
        file.write("%d " %frontal_landmark[i,1])
        file.write("0")
        if i != 67:
            file.write("\n")
    file.close()

    print(landmark_profile_path)
    file=open(landmark_profile_path,'w')
    for i in range(0,68):
        file.write("%d " %profile_landmark[i,0])
        file.write("%d " %profile_landmark[i,1])
        file.write("0")
        if i != 67:
            file.write("\n")
    file.close()
    
    frontal_drawn = draw_landmark(frontal, frontal_landmark)
    profile_drawn = draw_landmark(profile, profile_landmark)
    
    plt.imshow(frontal_drawn)
    plt.show()
    plt.imshow(profile_drawn)
    plt.show()
    
    return frontal_landmark, profile_landmark


    

img = "D:/current_project/RBF/RBF_Modeling/texturemap/IVCL_FaceData_renamed/image/001.jpg"
img_profile = "D:/current_project/RBF/RBF_Modeling/texturemap/IVCL_FaceData_renamed/images_profile/001.jpg"
frontal_landmark, profile_landmark = prediction(img, img_profile,"test_landmark.txt","test_landmark2.txt")
