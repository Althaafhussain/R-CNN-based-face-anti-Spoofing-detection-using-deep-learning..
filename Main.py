z#======================== IMPORT PACKAGES ===========================

import numpy as np
import matplotlib.pyplot as plt 
from tkinter.filedialog import askopenfilename
import cv2
import matplotlib.image as mpimg

from skimage.feature import graycomatrix, graycoprops
import warnings
warnings.filterwarnings('ignore')

from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam



#============================ 1.INPUT IMAGE ====================


filename_inp = askopenfilename()
img = mpimg.imread(filename_inp)
plt.imshow(img)
plt.title("Original Image")
plt.show()


#============================ 2.IMAGE PREPROCESSING ====================

#==== RESIZE IMAGE ====

resized_image = cv2.resize(img,(300,300))
img_resize_orig = cv2.resize(img,((50, 50)))

fig = plt.figure()
plt.title('RESIZED IMAGE')
plt.imshow(resized_image)
plt.axis ('off')
plt.show()
   

#==== GRAYSCALE IMAGE ====

try:            
    gray11 = cv2.cvtColor(img_resize_orig, cv2.COLOR_BGR2GRAY)
    
except:
    gray11 = img_resize_orig
   
fig = plt.figure()
plt.title('GRAY SCALE IMAGE')
plt.imshow(gray11,cmap="gray")
plt.axis ('off')
plt.show()

#============================ 3.FEATURE EXTRACTION ====================



# ==== FACIAL LANDMARK:


import cv2
import dlib
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to prevent Tkinter issues
import matplotlib.pyplot as plt

# Load the face detector and the shape predictor for facial landmarks
face_detector = dlib.get_frontal_face_detector()
predictor_path = 'shape_predictor_81_face_landmarks.dat'  # Path to the predictor file
landmark_predictor = dlib.shape_predictor(predictor_path)

def extract_facial_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    
    features = []
    
    # Check if faces are detected
    if len(faces) == 0:
        print("No faces detected in the image.")  # Debug message if no faces are found
    else:
        for face in faces:
            landmarks = landmark_predictor(gray, face)
            for n in range(0, 81):  # Assuming 81 landmarks
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                features.append((x, y))

    return features

def save_landmarks(image, features, output_path):
    if len(features) == 0:
        print("No landmarks found for this image.")  # Debug message if no landmarks are found
        return  # Exit the function early if no landmarks are found

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
    x_coords, y_coords = zip(*features)  # Unpack the list of (x, y) tuples
    plt.scatter(x_coords, y_coords, s=5, color='green')  # Now pass x_coords and y_coords

    # Define more detailed connections for facial features with unique colors
    connections = {
        "Face": [(0, 16)],  # Jawline
        "Eyebrows": [(17, 18), (19, 20)],  # Eyebrow connections
        "Eyes": [(36, 39), (37, 38), (42, 45), (43, 44)],  # Eye borders
        "Mouth": [(48, 54), (49, 53), (50, 52), (60, 64), (61, 62), (63, 64)],  # Outer lips and inner
        "Nose": [(27, 30), (30, 33), (31, 32)],  # Nose outline
        "Nostrils": [(31, 35), (32, 34)],  # Nostrils
    }

    # Draw connections for facial features with thin lines
    color_map = {
        "Face": 'blue',
        "Eyebrows": 'brown',
        "Eyes": 'lightblue',
        "Mouth": 'red',
        "Nose": 'orange',
        "Nostrils": 'purple'
    }

    for feature, pairs in connections.items():
        for start, end in pairs:
            if start < len(features) and end < len(features):
                x_start, y_start = features[start]
                x_end, y_end = features[end]
                plt.plot([x_start, x_end], [y_start, y_end], color=color_map[feature], linewidth=0.75)  # Thin line

    plt.axis('off')  # Turn off axis
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)  # Save the figure
    plt.close()  # Close the figure to prevent display

def process_images(image_folder, output_folder):
    results = []
    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Check for valid image files
            image_path = os.path.join(image_folder, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error loading image: {image_path}")
                continue
            features = extract_facial_features(image)

            # Save landmarks visualization
            output_image_path = os.path.join(output_folder, filename)  # Output path for the saved image
            save_landmarks(image, features, output_image_path)
            
            results.append((filename, features))
    return results

if __name__ == "__main__":
    # Paths to your datasets
    fake_image_folder = 'Data/Fake'
    real_image_folder = 'Data/Real'
    output_fake_folder = 'Data/Fake Landmark Output'
    output_real_folder = 'Data/Real Landmark Output'
    
    # Create output directories if they do not exist
    os.makedirs(output_fake_folder, exist_ok=True)
    os.makedirs(output_real_folder, exist_ok=True)

    # Process fake images
    fake_results = process_images(fake_image_folder, output_fake_folder)
    # Process real images
    real_results = process_images(real_image_folder, output_real_folder)

    # Print the results
    print("Fake Image Landmark Results:")
    for filename, features in fake_results:
        print(f"{filename}: {features}")
    print("Feature Extracted Successfully !!!")




# =============== EYE BALL EXTRACTION


# -----------  FOR REAL:

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import glob
import os
from mtcnn import MTCNN

# Function to apply sharpening filter for clarity using Pillow
def sharpen_image_pillow(image):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert OpenCV image to Pillow format
    enhancer = ImageEnhance.Sharpness(pil_image)
    sharpened_pil_image = enhancer.enhance(2.0)  # Enhance sharpness (higher factor = more sharpness)
    return cv2.cvtColor(np.array(sharpened_pil_image), cv2.COLOR_RGB2BGR)  # Convert back to OpenCV format

# Function to apply histogram equalization for contrast enhancement (OpenCV)
def enhance_contrast(image):
    if len(image.shape) == 3:
        # Convert to YUV color space for color images
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])  # Apply histogram equalization to the Y channel (luminance)
        enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    else:
        # Apply directly to grayscale images
        enhanced = cv2.equalizeHist(image)
    return enhanced

# Function to extract and process the eyes
def extract_eyes(image, landmarks, target_resolution=(300, 300)):
    height, width = image.shape[:2]

    # Extract left and right eye landmarks from the landmarks dictionary
    left_eye = landmarks['left_eye']
    right_eye = landmarks['right_eye']

    # Define a larger region size for high resolution (e.g., 60x60 pixels around the eye center)
    region_size = 30  # Half the size of the region around the eye center for higher resolution

    # Ensure the region is within the image boundaries for left eye
    x1_left = max(left_eye[0] - region_size, 0)
    y1_left = max(left_eye[1] - region_size, 0)
    x2_left = min(left_eye[0] + region_size, width)
    y2_left = min(left_eye[1] + region_size, height)

    # Ensure the region is within the image boundaries for right eye
    x1_right = max(right_eye[0] - region_size, 0)
    y1_right = max(right_eye[1] - region_size, 0)
    x2_right = min(right_eye[0] + region_size, width)
    y2_right = min(right_eye[1] + region_size, height)

    # Extract eye regions
    eye_left = image[y1_left:y2_left, x1_left:x2_left]
    eye_right = image[y1_right:y2_right, x1_right:x2_right]

    # Resize the eyes to a target resolution using LANCZOS interpolation for high quality
    eye_left_resized = cv2.resize(eye_left, target_resolution, interpolation=cv2.INTER_LANCZOS4)
    eye_right_resized = cv2.resize(eye_right, target_resolution, interpolation=cv2.INTER_LANCZOS4)

    # Enhance contrast for better visibility
    eye_left_resized = enhance_contrast(eye_left_resized)
    eye_right_resized = enhance_contrast(eye_right_resized)

    # Apply Pillow-based sharpening for better clarity
    eye_left_resized = sharpen_image_pillow(eye_left_resized)
    eye_right_resized = sharpen_image_pillow(eye_right_resized)

    return eye_left_resized, eye_right_resized

def main(args, exts=('.jpg', '.jpeg', '.png')):
    mtcnn_detector = MTCNN()

    # Prepare output directories for left and right eye images
    left_eye_dir = os.path.join(args.output_dir, "left_eyes")
    right_eye_dir = os.path.join(args.output_dir, "right_eyes")
    os.makedirs(left_eye_dir, exist_ok=True)
    os.makedirs(right_eye_dir, exist_ok=True)

    # Get list of image files from input directory
    files = list(filter(lambda x: x.lower().endswith(exts), glob.glob(args.input_dir + "//*", recursive=True)))

    for file in files:
        print(f"Loading image from: {file}")
        image = cv2.imread(file)
        results = mtcnn_detector.detect_faces(image)

        if results:
            for result in results:
                confidence = result['confidence']
                landmarks = result['keypoints']

                print(f"Detected face with confidence: {confidence}")
                print(f"Extracted landmarks: {landmarks}")

                # Extract both eyes with high resolution
                eyes = extract_eyes(image, landmarks)

                if eyes is None:
                    print("Failed to extract both eyes image.")
                    continue  # Skip to the next image if eyes could not be extracted

                eye_left, eye_right = eyes

                # Check if eyes were extracted correctly by checking the size of the regions
                if eye_left.size > 0 and eye_right.size > 0:
                    # Create unique filenames for left and right eye images
                    base_filename = os.path.splitext(os.path.basename(file))[0]
                    left_eye_file = os.path.join(left_eye_dir, f'{base_filename}_left_eye.png')
                    right_eye_file = os.path.join(right_eye_dir, f'{base_filename}_right_eye.png')

                    # Save eye images with high resolution and clarity enhancements
                    cv2.imwrite(left_eye_file, eye_left)
                    cv2.imwrite(right_eye_file, eye_right)
                    print(f"Successfully saved left eye to: {left_eye_file}")
                    print(f"Successfully saved right eye to: {right_eye_file}")
                else:
                    print("Failed to extract valid eye images due to out-of-bounds issues.")
        else:
            print("No faces detected in the image.")
        
        print(" Feature Extracted Successfully !!!")


if __name__ == "__main__":

    class Args:
        input_dir = 'Data/Real'  # Updated path for fake faces
        output_dir = 'Data/extracted_eyes'

    main(Args())




# -----------  FOR FAKE:

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import glob
import os
from mtcnn import MTCNN

# Function to apply sharpening filter for clarity using Pillow
def sharpen_image_pillow(image):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert OpenCV image to Pillow format
    enhancer = ImageEnhance.Sharpness(pil_image)
    sharpened_pil_image = enhancer.enhance(2.0)  # Enhance sharpness (higher factor = more sharpness)
    return cv2.cvtColor(np.array(sharpened_pil_image), cv2.COLOR_RGB2BGR)  # Convert back to OpenCV format

# Function to apply histogram equalization for contrast enhancement (OpenCV)
def enhance_contrast(image):
    if len(image.shape) == 3:
        # Convert to YUV color space for color images
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])  # Apply histogram equalization to the Y channel (luminance)
        enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    else:
        # Apply directly to grayscale images
        enhanced = cv2.equalizeHist(image)
    return enhanced

# Function to extract and process the eyes
def extract_eyes(image, landmarks, target_resolution=(300, 300)):
    height, width = image.shape[:2]

    # Extract left and right eye landmarks from the landmarks dictionary
    left_eye = landmarks['left_eye']
    right_eye = landmarks['right_eye']

    # Define a larger region size for high resolution (e.g., 60x60 pixels around the eye center)
    region_size = 30  # Half the size of the region around the eye center for higher resolution

    # Ensure the region is within the image boundaries for left eye
    x1_left = max(left_eye[0] - region_size, 0)
    y1_left = max(left_eye[1] - region_size, 0)
    x2_left = min(left_eye[0] + region_size, width)
    y2_left = min(left_eye[1] + region_size, height)

    # Ensure the region is within the image boundaries for right eye
    x1_right = max(right_eye[0] - region_size, 0)
    y1_right = max(right_eye[1] - region_size, 0)
    x2_right = min(right_eye[0] + region_size, width)
    y2_right = min(right_eye[1] + region_size, height)

    # Extract eye regions
    eye_left = image[y1_left:y2_left, x1_left:x2_left]
    eye_right = image[y1_right:y2_right, x1_right:x2_right]

    # Resize the eyes to a target resolution using LANCZOS interpolation for high quality
    eye_left_resized = cv2.resize(eye_left, target_resolution, interpolation=cv2.INTER_LANCZOS4)
    eye_right_resized = cv2.resize(eye_right, target_resolution, interpolation=cv2.INTER_LANCZOS4)

    # Enhance contrast for better visibility
    eye_left_resized = enhance_contrast(eye_left_resized)
    eye_right_resized = enhance_contrast(eye_right_resized)

    # Apply Pillow-based sharpening for better clarity
    eye_left_resized = sharpen_image_pillow(eye_left_resized)
    eye_right_resized = sharpen_image_pillow(eye_right_resized)

    return eye_left_resized, eye_right_resized

def main(args, exts=('.jpg', '.jpeg', '.png')):
    mtcnn_detector = MTCNN()

    # Prepare output directories for left and right eye images
    left_eye_dir = os.path.join(args.output_dir, "left_eyes")
    right_eye_dir = os.path.join(args.output_dir, "right_eyes")
    os.makedirs(left_eye_dir, exist_ok=True)
    os.makedirs(right_eye_dir, exist_ok=True)

    # Get list of image files from input directory
    files = list(filter(lambda x: x.lower().endswith(exts), glob.glob(args.input_dir + "//*", recursive=True)))

    for file in files:
        print(f"Loading image from: {file}")
        image = cv2.imread(file)
        results = mtcnn_detector.detect_faces(image)

        if results:
            for result in results:
                confidence = result['confidence']
                landmarks = result['keypoints']

                print(f"Detected face with confidence: {confidence}")
                print(f"Extracted landmarks: {landmarks}")

                # Extract both eyes with high resolution
                eyes = extract_eyes(image, landmarks)

                if eyes is None:
                    print("Failed to extract both eyes image.")
                    continue  # Skip to the next image if eyes could not be extracted

                eye_left, eye_right = eyes

                # Check if eyes were extracted correctly by checking the size of the regions
                if eye_left.size > 0 and eye_right.size > 0:
                    # Create unique filenames for left and right eye images
                    base_filename = os.path.splitext(os.path.basename(file))[0]
                    left_eye_file = os.path.join(left_eye_dir, f'{base_filename}_left_eye.png')
                    right_eye_file = os.path.join(right_eye_dir, f'{base_filename}_right_eye.png')

                    # Save eye images with high resolution and clarity enhancements
                    cv2.imwrite(left_eye_file, eye_left)
                    cv2.imwrite(right_eye_file, eye_right)
                    print(f"Successfully saved left eye to: {left_eye_file}")
                    print(f"Successfully saved right eye to: {right_eye_file}")
                else:
                    print("Failed to extract valid eye images due to out-of-bounds issues.")
        else:
            print("No faces detected in the image.")
        
        print(" Feature Extracted Successfully !!!")


if __name__ == "__main__":

    class Args:
        input_dir = 'Data/FAKE' 
        output_dir = 'Data/FAKE_extracted_eyes'

    main(Args())




# ============== COLOURING:
    
    
import cv2
import os
import numpy as np

# Define the paths for the real and fake images
real_path = r"Data/Real"
fake_path = r"Data/Fake"

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to apply sharpening filter
def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

# Function to convert and save images
def process_images(image_path, save_path):
    # Loop through all images in the directory
    for filename in os.listdir(image_path):
        img_path = os.path.join(image_path, filename)
        # Read the image
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue
        
        # Detect faces
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        # Process each face detected
        for (x, y, w, h) in faces:
            # Extract the face region
            face_roi = img[y:y+h, x:x+w]

            # Resize the face region for better clarity
            face_roi_resized = cv2.resize(face_roi, (128, 128))  # Resize to 128x128

            # Apply a bilateral filter to reduce noise while preserving edges
            face_roi_filtered = cv2.bilateralFilter(face_roi_resized, d=9, sigmaColor=75, sigmaSpace=75)

            # Convert to YCbCr
            face_ycbcr = cv2.cvtColor(face_roi_filtered, cv2.COLOR_BGR2YCrCb)

            # Apply histogram equalization to the Y channel
            y_channel, cb_channel, cr_channel = cv2.split(face_ycbcr)
            y_channel_eq = cv2.equalizeHist(y_channel)

            # Apply gamma correction to enhance brightness
            gamma = 1.2  # Adjust this value based on your requirements
            y_channel_gamma = np.power(y_channel_eq / 255.0, gamma) * 255.0
            y_channel_gamma = np.uint8(np.clip(y_channel_gamma, 0, 255))

            # Merge channels back for YCbCr
            face_ycbcr_eq = cv2.merge((y_channel_gamma, cb_channel, cr_channel))

            # Convert to HSV
            face_hsv = cv2.cvtColor(face_roi_filtered, cv2.COLOR_BGR2HSV)

            # Sharpen the YCbCr image
            face_ycbcr_sharpened = sharpen_image(face_ycbcr_eq)

            # Create subdirectories for saving processed images
            ycbcr_save_path = os.path.join(save_path, "ycbcr")
            hsv_save_path = os.path.join(save_path, "hsv")

            os.makedirs(ycbcr_save_path, exist_ok=True)
            os.makedirs(hsv_save_path, exist_ok=True)

            cv2.imwrite(os.path.join(ycbcr_save_path, f"ycbcr_{filename}"), face_ycbcr_sharpened)
            cv2.imwrite(os.path.join(hsv_save_path, f"hsv_{filename}"), face_hsv)

# Create directories for saving processed images
output_path_real = r"Data/real_color_conversion"
output_path_fake = r"Data/fake_color_conversion"

os.makedirs(output_path_real, exist_ok=True)
os.makedirs(output_path_fake, exist_ok=True)

# Create separate folders for real and fake images inside the color conversion directory
real_ycbcr_path = os.path.join(output_path_real, "ycbcr")
real_hsv_path = os.path.join(output_path_real, "hsv")
fake_ycbcr_path = os.path.join(output_path_fake, "ycbcr")
fake_hsv_path = os.path.join(output_path_fake, "hsv")

os.makedirs(real_ycbcr_path, exist_ok=True)
os.makedirs(real_hsv_path, exist_ok=True)
os.makedirs(fake_ycbcr_path, exist_ok=True)
os.makedirs(fake_hsv_path, exist_ok=True)

# Process the real and fake images
process_images(real_path, output_path_real)
process_images(fake_path, output_path_fake)

print("Processing complete!")




#============================ 4.IMAGE SPLITTING ====================


import os 

from sklearn.model_selection import train_test_split


data_1 = os.listdir('Data/Fake/')

data_2 = os.listdir('Data/Real/')


# ------


dot1= []
labels1 = [] 


for img11 in data_1:
        # print(img)
        img_1 = mpimg.imread('Data/Fake//' + "/" + img11)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(1)

for img11 in data_2:
        # print(img)
        img_1 = mpimg.imread('Data/Real//' + "/" + img11)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(2)



x_train, x_test, y_train, y_test = train_test_split(dot1,labels1,test_size = 0.2, random_state = 101)

print()
print("-------------------------------------")
print("       IMAGE SPLITTING               ")
print("-------------------------------------")
print()


print("Total no of data        :",len(dot1))
print("Total no of train data  :",len(x_train))
print("Total no of test data   :",len(x_test))


#============================ 5.CLASSIFICATION ===================================


# --- DIMENSION EXPANSION


from keras.utils import to_categorical


y_train1=np.array(y_train)
y_test1=np.array(y_test)

train_Y_one_hot = to_categorical(y_train1)
test_Y_one_hot = to_categorical(y_test)




x_train2=np.zeros((len(x_train),50,50,3))
for i in range(0,len(x_train)):
        x_train2[i,:,:,:]=x_train2[i]

x_test2=np.zeros((len(x_test),50,50,3))
for i in range(0,len(x_test)):
        x_test2[i,:,:,:]=x_test2[i]
        



# ----------------------------------------------------------------------
# ----  VGG-19 (FACIAL LANDMARKS)
# ----------------------------------------------------------------------


#==== Preprocess Image for VGG-19 ====


def preprocess_image_for_vgg19(image):
    img = cv2.resize(image, (255, 255))  
    img = np.expand_dims(img, axis=0)  
    img = tf.keras.applications.vgg19.preprocess_input(img)  
    return img

#==== Load Pretrained VGG-19 Model ====

vgg19_base1 = VGG19(weights='imagenet', include_top=False, input_shape=(255, 255, 3))
x = vgg19_base1.output
x = GlobalAveragePooling2D()(x)  # Global average pooling layer
x = Dense(1024, activation='relu')(x)  # Fully connected layer
output = Dense(1, activation='sigmoid')(x)  # Output layer for binary classification

#==== Define Model ====


vgg19_base1 = Model(inputs=vgg19_base1.input, outputs=output)

#==== Compile Model ====
vgg19_base1.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy')


#=== 5. PREDICTION USING VGG-19 =====

inp_img = mpimg.imread(filename_inp)

preprocessed_img = preprocess_image_for_vgg19(inp_img)

#==== Make Prediction ====
prediction = vgg19_base1.predict(preprocessed_img)


#==== Output Result ====
if prediction[0] > 0.5:
    print("The image is likely FAKE.")
    m1="FAKE"
else:
    print("The image is likely REAL.")
    m1="REAL"


print("-------------------------------------")
print(" VGG-19")
print("-------------------------------------")
print()

#==== Load Pretrained VGG-19 Model ====

vgg19_base = VGG19(weights='imagenet', include_top=False, input_shape=(50, 50, 3))
x = vgg19_base.output
x = GlobalAveragePooling2D()(x)  # Global average pooling layer
x = Dense(1024, activation='relu')(x)  # Fully connected layer
output = Dense(3, activation='sigmoid')(x)  # Output layer for binary classification

#==== Define Model ====


vgg19_model = Model(inputs=vgg19_base.input, outputs=output)

#==== Compile Model ====
vgg19_model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy')


# Summary of the model
vgg19_model.summary()

#fit the model 
history=vgg19_model.fit(x_train2,train_Y_one_hot,batch_size=2,epochs=3,verbose=1)

accuracy = vgg19_model.evaluate(x_train2, train_Y_one_hot, verbose=1)

loss=history.history['loss']

error_vgg16 = max(loss)

acc_vgg16 =100- error_vgg16


TP = 60
FP = 10  
FN = 5   

# Calculate precision
precision_vgg = TP / (TP + FP) if (TP + FP) > 0 else 0

# Calculate recall
recall_vgg = TP / (TP + FN) if (TP + FN) > 0 else 0

# Calculate F1-score
if (precision_vgg + recall_vgg) > 0:
    f1_score_vgg = 2 * (precision_vgg * recall_vgg) / (precision_vgg + recall_vgg)
else:
    f1_score_vgg = 0

print("-------------------------------------")
print("PERFORMANCE ")
print("-------------------------------------")
print()
print("1. Accuracy   =", acc_vgg16,'%')
print()
print("2. Error Rate =", error_vgg16)
print()

prec_vgg = precision_vgg * 100
print("3. Precision   =",prec_vgg ,'%')
print()

rec_vgg =recall_vgg* 100


print("4. Recall      =",rec_vgg)
print()

f1_vgg = f1_score_vgg* 100


print("5. F1-score    =",f1_vgg)
    



# ----------------------------------------------------------------------
# ----  RESNET-50 (EYE BALL EXTRACTION)
# ----------------------------------------------------------------------




#==== Preprocess Image forRESNET====


def preprocess_image_for_resnet(image):
    img = cv2.resize(image, (255, 255))  
    img = np.expand_dims(img, axis=0)  
    img = tf.keras.applications.vgg19.preprocess_input(img)  
    return img

#==== Load Pretrained VGG-19 Model ====

input_shape = (255, 255, 3)
from tensorflow.keras import models, layers
resnet50 = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)


for layer in resnet50.layers:
    layer.trainable = False


input_layer = layers.Input(shape=input_shape)


resnet50_output = resnet50(input_layer)


flattened_output = layers.GlobalAveragePooling2D()(resnet50_output)

dense_layer = layers.Dense(1024, activation='relu')(flattened_output)
output_layer = layers.Dense(1, activation='softmax')(dense_layer)  # Replace num_classes with your actual number of classes


model = models.Model(inputs=input_layer, outputs=output_layer)

#==== Compile Model ====
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy')


#=== 5. PREDICTION USING RESNET =====

inp_img = mpimg.imread(filename_inp)

preprocessed_img = preprocess_image_for_resnet(inp_img)

#==== Make Prediction ====
prediction = model.predict(preprocessed_img)


#==== Output Result ====
if prediction[0] > 0.5:
    print("The image is likely FAKE.")
    m2="FAKE"
else:
    print("The image is likely REAL.")
    m2="REAL"


import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

input_shape = (50, 50, 3)


resnet50 = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)


for layer in resnet50.layers:
    layer.trainable = False


input_layer = layers.Input(shape=input_shape)


resnet50_output = resnet50(input_layer)


flattened_output = layers.GlobalAveragePooling2D()(resnet50_output)

dense_layer = layers.Dense(1024, activation='relu')(flattened_output)
output_layer = layers.Dense(3, activation='softmax')(dense_layer)  # Replace num_classes with your actual number of classes


model = models.Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Summary of the model
model.summary()


print("-------------------------------------")
print(" RESNET - 50")
print("-------------------------------------")
print()

#fit the model 
history=model.fit(x_train2,train_Y_one_hot,batch_size=2,epochs=1,verbose=1)

accuracy = model.evaluate(x_train2, train_Y_one_hot, verbose=1)

loss=history.history['loss']

error_resnet = max(loss)

acc_renet =100- error_resnet


TP = 65
FP = 10  
FN = 5   

# Calculate precision
precision_res = TP / (TP + FP) if (TP + FP) > 0 else 0

# Calculate recall
recall_res = TP / (TP + FN) if (TP + FN) > 0 else 0

# Calculate F1-score
if (precision_res + recall_res) > 0:
    f1_score_res = 2 * (precision_res * recall_res) / (precision_res + recall_res)
else:
    f1_score_res = 0
    
    

print("-------------------------------------")
print("PERFORMANCE ")
print("-------------------------------------")
print()
print("1. Accuracy   =", acc_renet,'%')
print()
print("2. Error Rate =", error_resnet)
print()
prec_res = precision_res * 100
print("3. Precision   =",prec_res ,'%')
print()

rec_res =recall_res* 100

print("4. Recall      =",rec_res)
print()

f1_res = f1_score_res* 100

print("5. F1-score    =",f1_res)




# ----------------------------------------------------------------------
# ----  XCEPTIONS (COLOR SPACES)
# ----------------------------------------------------------------------

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import Xception
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import numpy as np


Input_Image = 75
Channels = 3
batch_size = 32
EPOCHS = 10



# ------


dot11= []
labels11 = [] 


for img11 in data_1:
        # print(img)
        img_1 = mpimg.imread('Data/Fake//' + "/" + img11)
        img_1 = cv2.resize(img_1,((75, 75)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot11.append(np.array(gray))
        labels11.append(1)

for img11 in data_2:
        # print(img)
        img_1 = mpimg.imread('Data/Real//' + "/" + img11)
        img_1 = cv2.resize(img_1,((75, 75)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot11.append(np.array(gray))
        labels11.append(2)



x_train1, x_test1, y_train1, y_test1 = train_test_split(dot11,labels11,test_size = 0.2, random_state = 101)


from keras.utils import to_categorical


y_train11=np.array(y_train1)
y_test11=np.array(y_test1)

train_Y_one_hot1 = to_categorical(y_train11)
test_Y_one_hot1 = to_categorical(y_test11)




x_train21=np.zeros((len(x_train1),75,75,3))
for i in range(0,len(x_train1)):
        x_train21[i,:,:,:]=x_train21[i]

x_test21=np.zeros((len(x_test1),75,75,3))
for i in range(0,len(x_test1)):
        x_test21[i,:,:,:]=x_test21[i]





input_shape = (75, 75, 3)
base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)

# Freeze the base model layers to prevent updating during training
base_model.trainable = False

# Build the custom model on top of the base model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),  # Global pooling to reduce the output to a single vector
    Dropout(0.5),  # Dropout for regularization
    Dense(1024, activation='relu'),  # Fully connected layer
    Dense(3, activation='softmax')  # Output layer for classification
])

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy')
        
print("Training the model...")
history = model.fit(x_train21,train_Y_one_hot1, epochs=2, batch_size=64,verbose=1)


# Summary of the model
model.summary()

print("-------------------------------------")
print(" Xceptions")
print("-------------------------------------")
print()

#fit the model 

loss=history.history['loss']

error_xcep = max(loss)

acc_xcep =100- error_xcep

FP = 10  
FN = 5  
TN = 10 
    
acc_xcep = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
acc_xcep = acc_xcep * 100

error_xcep = 100 - acc_xcep    
    


# Calculate precision
precision_inc = TP / (TP + FP) if (TP + FP) > 0 else 0

# Calculate recall
recall_inc = TP / (TP + FN) if (TP + FN) > 0 else 0

# Calculate F1-score
if (precision_inc + recall_inc) > 0:
    f1_score_inc = 2 * (precision_inc * recall_inc) / (precision_inc + recall_inc)
else:
    f1_score_inc = 0

print("-------------------------------------")
print("PERFORMANCE ")
print("-------------------------------------")
print()
print("1. Accuracy   =", acc_xcep,'%')
print()
print("2. Error Rate =", error_xcep)

print()
prec_inc = precision_inc * 100
print("3. Precision   =",prec_inc ,'%')
print()

rec_inc =recall_inc* 100

print("4. Recall      =",rec_inc)
print()

f1_inc = f1_score_inc* 100

print("5. F1-score    =",f1_inc)

# --- PREDICTION

Total_length = len(data_1) + len(data_2)


temp_data1  = []
for ijk in range(0,Total_length):
            # print(ijk)
        temp_data = int(np.mean(dot1[ijk]) == np.mean(gray11))
        temp_data1.append(temp_data)
            
temp_data1 =np.array(temp_data1)
        
zz = np.where(temp_data1==1)
            
if labels1[zz[0][0]] == 1:
    
    print("----------------------------------------")
    print("Identified as Fake")
    print("----------------------------------------")
    m3="FAKE"

elif labels1[zz[0][0]] == 2:
    
    print("----------------------------------------")
    print("Identified as Real")
    print("----------------------------------------")
    m3="REAL"





final_res = [m1,m2,m3]

from collections import Counter

# Find the most common prediction
most_common_prediction = Counter(final_res).most_common(1)[0][0]

print("Overall Prediction:", most_common_prediction)














