import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from scipy.special import softmax


def detect_and_draw_keypoints(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    keypoints = sorted(keypoints, key=lambda x: -x.response)
    keypoints = keypoints[:150]  
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)####
    return image_with_keypoints, keypoints
    

def load_images_and_matrix(folder_path):
    files = os.listdir(folder_path)
    image_files = [f for f in files if f.lower().endswith('.ppm')]
    images = [cv2.imread(os.path.join(folder_path, f)) for f in image_files]
    images = images[:2]
    # Load the matrix (assuming it's a text file containing the matrix)
    matrix_file = [f for f in files if f.lower().startswith('h')][0]
    matrix_path = os.path.join(folder_path, matrix_file)
    matrix = np.loadtxt(matrix_path)

    return images, matrix



def display_images_and_matrix(images, matrix):
    img0, key1= detect_and_draw_keypoints(images[0])
    img1, key2= detect_and_draw_keypoints(images[1])
   
    ig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(img0)
    axs[0].set_title('First Image')
    axs[0].axis('off')  # Hide axes ticks
    axs[1].imshow(img1)
    axs[1].set_title('Second Image')
    axs[1].axis('off')  # Hide axes ticks
    plt.tight_layout()
    plt.show()

    keypoint_coordinates1 = np.array([kp.pt for kp in key1])
    keypoint_coordinates2 = np.array([kp.pt for kp in key2])

    # Convert keypoints of the first image to homogeneous coordinates
    homogeneous_coordinates1 = np.hstack((keypoint_coordinates1, np.ones((keypoint_coordinates1.shape[0], 1))))

    # Apply the transformation matrix to the homogeneous coordinates of the first image
    transformed_coordinates1 = np.dot(matrix, homogeneous_coordinates1.T).T

    # Convert back to Cartesian coordinates by dividing by the third element (homogeneous coordinate)
    transformed_coordinates_xy1 = transformed_coordinates1[:, :2] / transformed_coordinates1[:, 2, None]

    # matched_pairs, M, notA, notB = match_keypoints(keypoint_coordinates1.tolist() ,transformed_coordinates_xy1.tolist(), keypoint_coordinates2.tolist())
    m, notA, notB = match_keypoints(transformed_coordinates_xy1.tolist(), keypoint_coordinates2.tolist())

   

# The greedy algorithm - not good
# def match_keypoints(keypoints1, keypoints2, max_distance = 3):
#     m = []
#     a = set()
#     b = set()
#     # Use the keypoints1 directly without reshaping
#     tree = KDTree(keypoints1, leaf_size=2)   
#     for j, x in enumerate(keypoints2):
#         # Reshape x to a 2D array with the same dimensionality as keypoints1
#         x_reshaped = np.array(x).reshape(1, -1)
#         dist, i = tree.query(x_reshaped, k=1)
#         if dist[0] < max_distance and int(i[0][0]) not in a:  # Convert to integer
#             m.append((int(i[0][0]), j))  # Convert to integer
#             a.add(int(i[0][0]))  # Convert to integer
#             b.add(j)

#     notA = set(range(len(keypoints1))) - a
#     notB = set(range(len(keypoints2))) - b

#     return m, notA, notB



def match_keypoints(keypoints1, keypoints2, max_distance = 3):
    m = []
    a = set()
    b = set()
    arrayDictA=[]
    arrayDictB=[]
    softmaxArrayDictA=[]
    softmaxArrayDictB=[]
    p = {}

    # Use the keypoints1 directly without reshaping
    treeA = KDTree(keypoints1, leaf_size=2)   
    treeB = KDTree(keypoints2, leaf_size=2)   

    for j, x in enumerate(keypoints2):
        # Reshape x to a 2D array with the same dimensionality as keypoints1
        x_reshaped = np.array(x).reshape(1, -1)
        dist, i = treeA.query(x_reshaped, k=5)
        
        # Initialize an empty dictionary
        result_dict = {}

        #Iterate through the indices and distances
        for q in range(5):
            if dist[0][q] <= max_distance:
                # Add each index-distance pair to the dictionary
                result_dict[i[0][q]] = -dist[0][q]
                
        # Store dictionary in array
        arrayDictB.append(result_dict)
        
        # Apply softmax to the values if the dictionary is not empty
        if result_dict:
            softmax_values = softmax(list(result_dict.values()))
            softmax_dict = {key: value for key, value in zip(result_dict.keys(), softmax_values)}
            softmaxArrayDictB.append(softmax_dict)
        else:
            softmaxArrayDictB.append({})
        
        
    for j, x in enumerate(keypoints1):
        # Reshape x to a 2D array with the same dimensionality as keypoints1
        x_reshaped = np.array(x).reshape(1, -1)
        dist, i = treeB.query(x_reshaped, k=5)
        
        # Initialize an empty dictionary
        result_dict = {}

        for q in range(5):
            if dist[0][q] <= max_distance:
                # Add each index-distance pair to the dictionary
                result_dict[i[0][q]] = -dist[0][q]
        
        # Store dictionary in array
        arrayDictA.append(result_dict)
        
        # Apply softmax to the values if the dictionary is not empty
        if result_dict:
            softmax_values = softmax(list(result_dict.values()))
            softmax_dict = {key: value for key, value in zip(result_dict.keys(), softmax_values)}
            softmaxArrayDictA.append(softmax_dict)
        else:
            softmaxArrayDictA.append({})
        
        
    #choose the high value in p
    for i in range (len(softmaxArrayDictA)):
        for j in range (len(softmaxArrayDictB)):
            if softmaxArrayDictB[j].get(i) and softmaxArrayDictA[i].get(j):
                p[(i,j)] = softmaxArrayDictB[j].get(i)*softmaxArrayDictA[i].get(j)
    sorted_p = dict(sorted(p.items(), key=lambda item: item[1]))
    threshhold = 0.25
    
    for key, value in sorted_p.items():
        i = key[0] 
        j = key[1]  
        if value >= threshhold and i not in a and j not in b:
            m.append((i,j))
            a.add(i)
            b.add(j)
           
       


    # print("ARRAY A", softmaxArrayDictA)
    # print("ARRAY B", softmaxArrayDictB)
    print("sorted_p ", sorted_p)
    print("len of sorted_p: ", len(sorted_p))

    notA = set(range(len(keypoints1))) - a
    notB = set(range(len(keypoints2))) - b
    print("notA ", notA)
    print("len of notA: ", len(notA))
    print("notB ", notB)
    print("len of notB: ", len(notB))
    print("m ", m)
    print("len of m: ", len(m))
    return m, notA, notB


hpatches_dataset_path = r'C:\Users\asyer\Desktop\final\finalProject\hp\hpatches-sequences-release'

# Iterate through folders in the dataset
for folder_name in os.listdir(hpatches_dataset_path):
    if(folder_name.lower().startswith('v')):
        folder_path = os.path.join(hpatches_dataset_path, folder_name)

        # Check if it's a directory
        if os.path.isdir(folder_path):
            images, matrix = load_images_and_matrix(folder_path)

            display_images_and_matrix(images, matrix)

            # Wait for a key press before moving to the next folder
            cv2.waitKey(0)
            cv2.destroyAllWindows()





