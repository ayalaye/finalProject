import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        images, matrix = self.data[idx]
        # Convert images and matrix to tensors
        # images = [torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32) for image in images]  # Convert to CHW format
        # matrix = torch.tensor(matrix, dtype=torch.float32)
        m, notA,notB, key1, key2, des1, des2 = display_images_and_matrix(images,matrix)
        
        return {'keypoints1': key1, 'descriptors1': des1, 'keypoints2': key2, 'descriptors2': des2, 'notA': notA, 'notB': notB, 'm': m}


def load_images_and_matrix(folder_path, i, j):
    if i==1:
        pattern = "H_"+str(i)+"_"+str(j)
        matrix_path = os.path.join(folder_path, pattern)
        matrix = np.loadtxt(matrix_path)
    elif j==1:
        pattern = "H_"+str(j)+"_"+str(i)
        matrix_path = os.path.join(folder_path, pattern)
        matrix = np.loadtxt(matrix_path)
        matrix = np.linalg.inv(matrix)
    else:
        pattern1 = "H_"+str(1)+"_"+str(i)
        pattern2 = "H_"+str(1)+"_"+str(j)
        matrix_path1 = os.path.join(folder_path, pattern1)
        matrix_path2 = os.path.join(folder_path, pattern2)
        matrix1 = np.loadtxt(matrix_path1)
        matrix2 = np.loadtxt(matrix_path2)
        matrix = np.linalg.inv(matrix1) @ matrix2

    files = os.listdir(folder_path)
    image_files = [f for f in files if f.lower().endswith('.ppm')]

    # Get the file names for images i and j
    selected_images = [image_files[i - 1], image_files[j - 1]]

    # Read only the selected images
    images = [cv2.imread(os.path.join(folder_path, f)) for f in selected_images]

    return images, matrix



def detect_and_draw_keypoints(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    keypoints = sorted(keypoints, key=lambda x: -x.response)
    keypoints = keypoints[:150]  
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)####
    return image_with_keypoints, keypoints, descriptors
   


def match_keypoints(keypoints1, keypoints2, max_distance = 3):
    m = []
    a = set()
    b = set()
    # Use the keypoints1 directly without reshaping
    tree = KDTree(keypoints1, leaf_size=2)   
    for j, x in enumerate(keypoints2):
        # Reshape x to a 2D array with the same dimensionality as keypoints1
        x_reshaped = np.array(x).reshape(1, -1)
        dist, i = tree.query(x_reshaped, k=1)
        if dist[0] < max_distance and int(i[0][0]) not in a:  # Convert to integer
            m.append((int(i[0][0]), j))  # Convert to integer
            a.add(int(i[0][0]))  # Convert to integer
            b.add(j)

    notA = set(range(len(keypoints1))) - a
    notB = set(range(len(keypoints2))) - b

    return m, notA, notB



def display_images_and_matrix(images, matrix):
    img0, key1, des1= detect_and_draw_keypoints(images[0])
    img1, key2, des2= detect_and_draw_keypoints(images[1])
   
    ig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(img0)
    axs[0].set_title('First Image')
    axs[0].axis('off')  # Hide axes ticks
    axs[1].imshow(img1)
    axs[1].set_title('Second Image')
    axs[1].axis('off')  # Hide axes ticks
    plt.tight_layout()
    plt.show()
    print(matrix)
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

    return m, notA,notB, key1, key2, des1, des2


count=0
hpatches_dataset_path = r'C:\Users\asyer\Desktop\final\finalProject\hp\hpatches-sequences-release'
dic=[]
# Iterate through folders in the dataset
for folder_name in os.listdir(hpatches_dataset_path):
    count+=1

    folder_path = os.path.join(hpatches_dataset_path, folder_name)

    # Check if it's a directory
    if os.path.isdir(folder_path):
        for i in range(1, 7):
            for j in range(1, 7):
                if i!=j: ## contain 2-3 and also 3-2
                    images, matrix = load_images_and_matrix(folder_path, i , j)
                    dic.append((images,matrix))
                    # display_images_and_matrix(images, matrix)
                    print("--")
                    # Wait for a key press before moving to the next folder
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Create an instance of the custom dataset class
dataset = MyDataset(dic)
print(count)
# Example usage:
# Accessing the first sample in the dataset
# dicti = dataset[0]
# print("dicti:", dicti)

# Get the length of the dataset
dataset_length = len(dataset)
print("Length of the dataset:", dataset_length)
