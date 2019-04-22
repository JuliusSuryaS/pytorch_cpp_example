import os, sys
import face_alignment
from skimage import io
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

def predict_and_save(frontal, profile, landmark_path, landmark_profile_path):
    print(frontal, profile)
    print(landmark_path, landmark_profile_path)
    # Run the 3D face alignment on a test image, without CUDA.
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, enable_cuda=False, flip_input=False,use_cnn_face_detector=True)
    frontal_img = io.imread(frontal)
    profile_img = io.imread(profile)
    frontal_landmark = fa.get_landmarks(frontal_img)[-1]
    profile_landmark = fa.get_landmarks(profile_img)[-1]
    #print(preds[0,0])
    #print(preds[0,1])
    #print(preds[67,0])
    #write to txt

def save_landmark(landmark, landmark_path):
    print(landmark_path)
    file=open(landmark_path,'w')
    for i in range(0,68):
        file.write("%d " %landmark[i,0])
        file.write("%d " %landmark[i,1])
        file.write("0")
        if i != 67:
            file.write("\n")
    file.close()


base_path = "D:/current_project/RBF/RBF_Modeling/texturemap/"
frontal_path = base_path + "IVCL_FaceData_renamed/image/"
profile_path = base_path + "IVCL_FaceData_renamed/images_profile/"
landmark_frontal_path = base_path + "IVCL_FaceData_renamed/landmarks_dnn/"
landmark_profile_path = base_path + "IVCL_FaceData_renamed/landmarks_dnn_profile/"

detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, enable_cuda=False, flip_input=False,use_cnn_face_detector=True)

for root, dirs, files in os.walk(frontal_path):
    for name in files:
        base_name = name.split(".")
        frontal_img_path = os.path.join(frontal_path, name)
        profile_img_path = os.path.join(profile_path, name)
        landmark_save = os.path.join(landmark_frontal_path, base_name[0] + '.txt')
        landmark_profile_save = os.path.join(landmark_profile_path, base_name[0] + '.txt')

        ## Facial landmark detection
        try:
            img_front = io.imread(frontal_img_path)
            img_prof = io.imread(profile_img_path)

            land_front = detector.get_landmarks(img_front)[-1]
            land_prof = detector.get_landmarks(img_prof)[-1]
            #save_landmark(land_front, landmark_save)
            #save_landmark(land_prof, landmark_profile_save)
        except:
            print("Error")
        input('waiting for next input.......')


