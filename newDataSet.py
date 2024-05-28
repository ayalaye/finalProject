import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import torch
from torch.utils.data import Dataset
from scipy.special import softmax
 
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        images, matrix = self.data[idx]
        m, notA,notB, key1, key2, des1, des2 = match(images,matrix)
        print("len des1 :",des1.shape)
        print("len des2 :",des2.shape)

        return {'key1': key1, 'des1': des1, 'key2': key2, 'des2': des2, 'notA': notA, 'notB': notB, 'm': m}


def load_images_and_matrix(folder_path):
    

    image_files = os.listdir(folder_path)
    random_number = random.randint(0, len(image_files) - 1)

    # Get the file names for images i and j
    img = cv2.imread(os.path.join(folder_path, image_files[random_number])) 

    rho = 32
    patch_size = 128
    top_point = (32, 32)
    left_point = (patch_size + 32, 32)
    bottom_point = (patch_size + 32, patch_size + 32)
    right_point = (32, patch_size + 32)
    four_points = [top_point, left_point, bottom_point, right_point]

    perturbed_four_points = []
    for point in four_points:
        perturbed_four_points.append(
            (point[0] + random.randint(-rho, rho), point[1] + random.randint(-rho, rho)))

    H = cv2.getPerspectiveTransform(np.float32(perturbed_four_points), np.float32(four_points))
    warped_image = cv2.warpPerspective(img, H, (320, 240))


    # Read only the selected images
    images = [img, warped_image]

    return images, H



def detect_and_draw_keypoints(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    # keypoints = sorted(keypoints, key=lambda x: -x.response)
    keypoints = keypoints[:600]  
    descriptors = descriptors[:600]
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)####
    return image_with_keypoints, keypoints, descriptors


def draw_matching_lines(image1, keypoints1, image2, keypoints2, matches):
    # Create a new image by concatenating the two input images side by side
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    vis[:h1, :w1] = image1
    vis[:h2, w1:w1 + w2] = image2

    # Draw lines between matched keypoints
    for match in matches:
        # Get the matched keypoints
        kp1 = keypoints1[match[0]]
        kp2 = keypoints2[match[1]]

        # Offset the x-coordinate of keypoints in the second image by the width of the first image
        kp2.pt = (kp2.pt[0] + w1, kp2.pt[1])
        color = tuple(np.random.randint(0, 255, 3).tolist())

        # Draw a line between the matched keypoints
        cv2.line(vis, (int(kp1.pt[0]), int(kp1.pt[1])), (int(kp2.pt[0]), int(kp2.pt[1])), color, 1)

    return vis


def match_keypoints(keypoints1, keypoints2, max_distance = 3):
    m = []
    a = set()
    b = set()

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
        # if len(i[0]) < 5:
        #     return dataset[random.randint(0, 99)]
        # Initialize an empty dictionary
        result_dict = {}

        #Iterate through the indices and distances
        for q in range(len(dist[0])): 
            if dist[0][q] <= max_distance: 
                # Add each index-distance pair to the dictionary
                result_dict[i[0][q]] = -dist[0][q]
                
        
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

    
    for key, value in sorted_p.items():
        i = key[0] 
        j = key[1]  
        if value >= 0.25 and i not in a and j not in b:
            m.append((i,j))
            a.add(i)
            b.add(j)
           

    notA = set(range(len(keypoints1))) - a
    notB = set(range(len(keypoints2))) - b
    
    print("len of sorted_p: ", len(sorted_p))
    print("len of notA: ", len(notA))
    print("len of notB: ", len(notB))
    print("len of m: ", len(m))
    return m, notA, notB



def match(images, matrix):
    img0, key1, des1= detect_and_draw_keypoints(images[0])
    img1, key2, des2= detect_and_draw_keypoints(images[1])
   
    ig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(images[0])
    axs[0].set_title('First Image')
    axs[0].axis('off')  # Hide axes ticks
    axs[1].imshow(images[1])
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

    m, notA, notB = match_keypoints(transformed_coordinates_xy1.tolist(), keypoint_coordinates2.tolist())

    # Draw matching lines on the images
    matched_image = draw_matching_lines(img0, key1, img1, key2, m)

    # Display the result
    plt.figure(figsize=(10, 5))
    plt.imshow(matched_image)
    plt.axis('off')
    plt.tight_layout()

    plt.show()

    return m, notA,notB, key1, key2, des1, des2

#main:
# count=0
# data_path = './resize_photos'
# dic=[]
# # Iterate through folders in the dataset
# for i in range(100):
#     images, matrix = load_images_and_matrix(data_path)
#     dic.append((images,matrix))
    

# # Create an instance of the custom dataset class
# dataset = MyDataset(dic)
# # Example usage:
# # Accessing the first sample in the dataset
# for i in range(100):
#     print("dicti:", dataset[i])

# # Get the length of the dataset
# dataset_length = len(dataset)
# print("Length of the dataset:", dataset_length)





