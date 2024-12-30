import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import torch
from torch.utils.data import Dataset
from scipy.special import softmax
import random
import json 
import time

class MyDataset(Dataset):
    def __init__(self, data):
        # Creation of the data - happens only once
        
        # Check if the file exists 
        if not os.path.exists('/storage/shared/data_ayala_elisheva/data/dataOxfordPets20K.json.json'):
            data_path = data
            dic=[]
            
            # Iterate through folders in the dataset
            with open('/storage/shared/data_ayala_elisheva/data/dataOxfordPets20K.json', "w") as file:
                for i in range(20000):
                    print(i)
                    images, matrix = load_images_and_matrix(data_path)
                    result=match(images,matrix)
                    if result==0:
                        continue
                    print(i)
                    m, notA, notB, key1, key2, des1, des2 = result                    
                    key1 = list([(kp.pt[0], kp.pt[1]) for kp in key1])
                    key2 = list([(kp.pt[0], kp.pt[1]) for kp in key2])
                    
                    des1 = list([row.tolist() for row in des1])
                    des2 = list([row.tolist() for row in des2])
                    
                    notA = list(notA)
                    notB = list(notB)
                    
                    m = list([(i[0],i[1]) for i in m])
                    
                    dic=({'key1': key1, 'des1': des1, 'key2': key2, 'des2': des2, 'notA': notA, 'notB': notB, 'm': m})
                    # Save dictionary to JSON file 
                    
                    json.dump(dic, file)
                    file.write('\n')

        # self.infile1 = open('data/data1__.json', 'r')
        # self.infile2 = open('data/data2.json', 'r')
        # self.infile3 = open('data/data3.json', 'r')
        # self.infile4 = open('data/data4.json', 'r')
        # self.infile5 = open('data/data5.json', 'r')
        # self.infile6 = open('data/data6.json', 'r')
        # self.infile7 = open('data/data7.json', 'r')
        # self.infile8 = open('data/data8.json', 'r')
        # self.infile9 = open('data/data9.json', 'r')
        # self.infile10 = open('data/data10.json', 'r')
        # self.infileExample = open('data/data_example.json', 'r')
        # self.infileOxfordBuildings20K = open('data/dataOxfordBuildings20K.json', 'r')
        # self.infile_big_data = open('data/big_data.json', 'r')

        self.infileOxfordPets20K = open('data/dataOxfordPets20K.json', 'r')
        
    # Return length of dataset 
    def __len__(self):
        return 20000

    #  Return the data for a single pair of images
    def __getitem__(self, idx):
        # self.infile1.seek(index_data[idx]['offset'])
        # line = self.infile1.readline()
        # data = json.loads(line)
        # if idx < 10000:
        #     self.infile1.seek(index_data[idx]['offset'])
        #     line = self.infile1.readline()
        #     data = json.loads(line)
        # elif 10000<=idx<20000:
        #     idx=idx-10000
        #     self.infile2.seek(index_data2[idx]['offset'])
        #     line = self.infile2.readline()
        #     data = json.loads(line)
        # elif 20000<=idx<30000:
        #     idx=idx-20000
        #     self.infile3.seek(index_data3[idx]['offset'])
        #     line = self.infile3.readline()
        #     data = json.loads(line)
        # elif 30000<=idx<40000:
        #     idx=idx-30000
        #     self.infile4.seek(index_data4[idx]['offset'])
        #     line = self.infile4.readline()
        #     data = json.loads(line)
        # elif 40000<=idx<50000:
        #     idx=idx-40000
        #     self.infile5.seek(index_data5[idx]['offset'])
        #     line = self.infile5.readline()
        #     data = json.loads(line)
        # elif 50000<=idx<60000:
        #     idx=idx-50000
        #     self.infile6.seek(index_data6[idx]['offset'])
        #     line = self.infile6.readline()
        #     data = json.loads(line)
        # elif 60000<=idx<70000:
        #     idx=idx-60000
        #     self.infile7.seek(index_data7[idx]['offset'])
        #     line = self.infile7.readline()
        #     data = json.loads(line)
        # elif 70000<=idx<80000:
        #     idx=idx-70000
        #     self.infile8.seek(index_data8[idx]['offset'])
        #     line = self.infile8.readline()
        #     data = json.loads(line)
        # elif 80000<=idx<90000:
        #     idx=idx-80000
        #     self.infile9.seek(index_data9[idx]['offset'])
        #     line = self.infile9.readline()
        #     data = json.loads(line)
        # elif 90000<=idx<100000:
        #     idx=idx-90000
        #     self.infile10.seek(index_data10[idx]['offset'])
        #     line = self.infile10.readline()
        #     data = json.loads(line)
        # elif 100000<=idx<120000:
        #     idx=idx-100000

        # self.infileOxfordPets20K.seek(index_dataOxfordPets20K[idx]['offset'])
        # line = self.infileOxfordPets20K.readline()
        # data = json.loads(line)
        
        # self.infile_big_data.seek(index_big_data[idx]['offset'])
        # line = self.infile_big_data.readline()
        # data = json.loads(line)  
        data = json.load(self.infileOxfordPets20K)
        return data
        
       
def load_images_and_matrix(folder_path):
    """
    Loads a random image from a specified folder, generates a warped version 
    of the image using a perspective transformation, and returns both the 
    original and warped images along with the transformation matrix.
    """

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
    """
    Detects keypoints and computes descriptors using the SIFT algorithm,
    and draws the keypoints on the image
    """
    image_with_keypoints=0
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    
    if descriptors is None:
        keypoints=[]
        descriptors=[]
        return image_with_keypoints,keypoints , descriptors

    # keypoints = sorted(keypoints, key=lambda x: -x.response)
    keypoints = keypoints[:600]  
    descriptors = descriptors[:600]
    # image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)####
    
    return image_with_keypoints, keypoints, descriptors


def draw_matching_lines(image1, keypoints1, image2, keypoints2, matches):
   """ Draws lines between matched keypoints from two images"""
    
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
        # cv2.line(vis, (int(kp1.pt[0]), int(kp1.pt[1])), (int(kp2.pt[0]), int(kp2.pt[1])), color, 1)

    return vis


def match_keypoints(keypoints1, keypoints2, max_distance = 3):
    """
    Matches keypoints between two sets of keypoints using a KD-Tree for efficient nearest-neighbor 
    search. The function calculates softmax values for the distance-based similarities between 
    keypoints and returns the best matches based on a threshold.
    """
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

    return m, notA, notB


def match(images, matrix):
    """
    Finds matching keypoints between two images after applying a geometric transformation 
    to the first image using a provided transformation matrix. The function detects keypoints 
    in both images, applies the transformation to the keypoints in the first image, and 
    then matches the transformed keypoints to the keypoints in the second image.
    """
    img0, key1, des1= detect_and_draw_keypoints(images[0])
    img1, key2, des2= detect_and_draw_keypoints(images[1])
    if len(key1)<5 or len(key2)<5 :
        return 0

    # ig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # axs[0].imshow(images[0])
    # axs[0].set_title('First Image')
    # axs[0].axis('off')  # Hide axes ticks
    # axs[1].imshow(images[1])
    # axs[1].set_title('Second Image')
    # axs[1].axis('off')  # Hide axes ticks
    # plt.tight_layout()
    # plt.show()

    keypoint_coordinates1 = np.array([kp.pt for kp in key1])
    keypoint_coordinates2 = np.array([kp.pt for kp in key2])

    # Convert keypoints of the first image to homogeneous coordinates
    homogeneous_coordinates1 = np.hstack((keypoint_coordinates1, np.ones((keypoint_coordinates1.shape[0], 1))))

    # Apply the transformation matrix to the homogeneous coordinates of the first image
    transformed_coordinates1 = np.dot(matrix, homogeneous_coordinates1.T).T

    # Convert back to Cartesian coordinates by dividing by the third element (homogeneous coordinate)
    transformed_coordinates_xy1 = transformed_coordinates1[:, :2] / transformed_coordinates1[:, 2, None]
   
    m, notA, notB = match_keypoints(transformed_coordinates_xy1.tolist(), keypoint_coordinates2.tolist())

    ## Draw matching lines on the images
    # matched_image = draw_matching_lines(img0, key1, img1, key2, m)

    ## Display the result
    # plt.figure(figsize=(10, 5))
    # plt.imshow(matched_image)
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()
   
    return m, notA,notB, key1, key2, des1, des2


def create_index(input_file, index_file):
     """
    Creates an index for a data file by storing the offset of each line in the file.
    The index is saved as a JSON file, where each entry contains the index and the byte offset 
    of the corresponding line in the original data file.
    """
    with open(input_file, 'r') as infile, open(index_file, 'w') as outfile:
        index = []
        offset = 0
        for line in infile:
            index.append({'index': len(index), 'offset': offset})
            offset += (len(line))
        json.dump(index, outfile)



# data_path='/storage/shared/data_ayala_elisheva/data/dataOxfordPets20K.json' 
# dataset = MyDataset(data_path)

with open('indexes/indexOxfordPets20K.json', 'r') as infile:
    index_dataOxfordPets20K = json.load(infile)

# with open('indexes/index_big_data.json', 'r') as infile:
#     index_big_data= json.load(infile)
# with open('indexes/index.json', 'r') as infile:
#     index_data = json.load(infile)
# with open('indexes/index2.json', 'r') as infile:
#     index_data2 = json.load(infile)
# with open('indexes/index3.json', 'r') as infile:
#     index_data3 = json.load(infile)
# with open('indexes/index4.json', 'r') as infile:
#     index_data4 = json.load(infile)
# with open('indexes/index5.json', 'r') as infile:
#     index_data5 = json.load(infile)
# with open('indexes/index6.json', 'r') as infile:
#     index_data6 = json.load(infile)
# with open('indexes/index7.json', 'r') as infile:
#     index_data7 = json.load(infile)
# with open('indexes/index8.json', 'r') as infile:
#     index_data8 = json.load(infile)
# with open('indexes/index9.json', 'r') as infile:
#     index_data9 = json.load(infile)
# with open('indexes/index10.json', 'r') as infile:
#     index_data10 = json.load(infile)
# with open('indexes/indexOxfordBuildings20K.json', 'r') as infile:
#     index_dataOxfordBuildings20K = json.load(infile)
