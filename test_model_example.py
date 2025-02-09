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
import torch
from GNN import LightGlue 
from torch.utils.data import DataLoader,Subset,random_split
from newDataSet import *

"""testing the model after training and saving"""

   
def loss_(P, M, not_A, not_B, sigma_A, sigma_B):
    epsilon = 1e-8
    M = torch.tensor(M, dtype=torch.long)
    not_A = torch.tensor(not_A, dtype=torch.long)
    not_B = torch.tensor(not_B, dtype=torch.long)

    if P.dim() == 1:
        correspondence_loss_sum = (torch.log(P[M] + epsilon).sum()) / len(M)
       
    else:
        correspondence_loss_sum = (torch.log(P[M[:, 0], M[:, 1]] + epsilon).sum()) / len(M)

    unmatchable_loss_A = 0.5 * (torch.log(1 - sigma_A[not_A] + epsilon)).sum() / len(not_A)
    unmatchable_loss_B = 0.5 * (torch.log(1 - sigma_B[not_B] + epsilon)).sum() / len(not_B)
   
    loss = correspondence_loss_sum + unmatchable_loss_A + unmatchable_loss_B
    return -loss


def test(model, loader, threshold):
    model.eval()  
    total_loss = 0
    total_iou = 0
    num_samples = 0

    with torch.no_grad():
        for data in loader.dataset:
            if len(data['m']) == 0:
                continue
           
            xA = torch.tensor(data['des1'], dtype=torch.float32)
            xB = torch.tensor(data['des2'], dtype=torch.float32)

            key1 = torch.tensor(data['key1'], dtype=torch.float32)
            key2 = torch.tensor(data['key2'], dtype=torch.float32)

            sigma_A, sigma_B, P = model(xA, xB,key1,key2)

            M = torch.zeros_like(P)
            for match in data['m']:
                M[match[0], match[1]] = 1

            intersection = torch.sum((P > threshold) & (M == 1)).item()
            union = torch.sum((P > threshold) | (M == 1)).item()

            iou = intersection / union if union > 0 else 0
            print("accuracy ", iou)
            total_iou += iou
           
            loss = loss_(P, data['m'], data['notA'], data['notB'], sigma_A, sigma_B)
            print("loss ",loss)
            total_loss += loss.item()
           
            num_samples += 1
           

    avg_loss = total_loss / len(loader.dataset)
    avg_accuracy = total_iou / num_samples if num_samples > 0 else 0
    # print("acc:"+ avg_accuracy)
    return avg_loss, avg_accuracy  


dataset = MyDataset("")

input_dim = 128  
output_dim = 128  
nhead=2
num_layers=2

model = LightGlue(input_dim, output_dim, nhead, num_layers)
model.load_state_dict(torch.load('petsGraphs/model_2024.12.08.07.37.46.pt'), strict=False)

lr=0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
threshold=0.25

print("-------------------------------------------testing-------------------------------------")
# test_dataset=Subset(dataset, range(30))

test_loader = DataLoader(dataset, batch_size=1, shuffle=False,num_workers=16)
test_loss, test_accuracy = test(model, test_loader, threshold)
print("avg_loss ",test_loss)
print("avg acc ",test_accuracy)



# input_dim = 128
# output_dim = 128
# nhead = 2
# num_layers = 2
# with open('indexes/indexOxfordPets20K.json', 'r') as infile:
#     index_dataOxfordPets20K = json.load(infile)

# # with open('data/dataOxfordPets20K.json', 'r') as file:
# #     file.seek(0)
# #     line = file.readline()
# #     data = json.loads(line)
# # with open('data/data_Abyssinian_9.json', 'r') as file:
# #     data = json.load(file)
# # טוענים את המודל
# loaded_model = LightGlue(input_dim, output_dim, nhead, num_layers)
# loaded_model.load_state_dict(torch.load('/storage/shared/data_ayala_elisheva/savedModels/model_2024.11.26.06.53.50.pt'), strict=False)
# loaded_model.eval()

# # loaded_model = torch.load('/storage/shared/data_ayala_elisheva/savedModels/model_2024.11.26.06.53.50.pt')
# # loaded_model.eval()



# # for param in loaded_model.parameters():
# #     print(param)

# with open('data/dataOxfordPets20K.json', 'r') as file:
#     for i in range(100):
#         file.seek(index_dataOxfordPets20K[i]['offset'])
#         line = file.readline()
#         data = json.loads(line)
#         xA = torch.tensor(data.get('des1'), dtype=torch.float32)
#         xB = torch.tensor(data.get('des2'), dtype=torch.float32)
#         posA = torch.tensor(data.get('key1'), dtype=torch.float32)
#         posB = torch.tensor(data.get('key2'), dtype=torch.float32)
        

#         # מבצעים את האינפרנס
#         with torch.no_grad():
#             sigma_A, sigma_B, P = loaded_model(xA, xB, posA, posB)
#             # print(len(data.get('m')))
#             # print(len(P))
           
#             M = torch.zeros_like(P)
#             for match in data['m']:
#                 M[match[0], match[1]] = 1

#             intersection = torch.sum((P > 0.25) & (M == 1)).item()         
#             # print(intersection)
#             union = torch.sum((P > 0.25) | (M == 1)).item()
#             # print(union)
#             iou = intersection / union if union > 0 else 0
#             print(iou)  
#             # print(data.get('m'))
        


# def load_images_and_matrix(folder_path):
#     start=time.time()


#     # Get the file names for images i and j
#     img = cv2.imread(folder_path)
    
#     rho = 32
#     patch_size = 128
#     top_point = (32, 32)
#     left_point = (patch_size + 32, 32)
#     bottom_point = (patch_size + 32, patch_size + 32)
#     right_point = (32, patch_size + 32)
#     four_points = [top_point, left_point, bottom_point, right_point]

#     perturbed_four_points = []
#     for point in four_points:
#         perturbed_four_points.append(
#             (point[0] + random.randint(-rho, rho), point[1] + random.randint(-rho, rho)))

#     H = cv2.getPerspectiveTransform(np.float32(perturbed_four_points), np.float32(four_points))
#     warped_image = cv2.warpPerspective(img, H, (320, 240))
#     plt.imsave("warped.png",warped_image)
#     # Read only the selected images
#     images = [img, warped_image]

    

#     end= time.time()
#     # print("load_images_and_matrix time is",end-start)
#     return images, H

# def detect_and_draw_keypoints(image):
#     if image is None:
#         print("none")
#     sift = cv2.SIFT_create()
#     keypoints, descriptors = sift.detectAndCompute(image, None)
#     print(keypoints)
#     print(descriptors)
#     # keypoints = sorted(keypoints, key=lambda x: -x.response)
#     keypoints = keypoints[:600]  
#     descriptors = descriptors[:600]
#     image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)####
#     # image_with_keypoints=0
#     return image_with_keypoints, keypoints, descriptors

# def draw_matching_lines(image1, keypoints1, image2, keypoints2, matches):
#     # Create a new image by concatenating the two input images side by side
#     h1, w1 = image1.shape[:2]
#     h2, w2 = image2.shape[:2]
#     vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
#     vis[:h1, :w1] = image1
#     vis[:h2, w1:w1 + w2] = image2

#     # Draw lines between matched keypoints
#     for match in matches:
#         # Get the matched keypoints
#         kp1 = keypoints1[match[0]]
#         kp2 = keypoints2[match[1]]

#         # Offset the x-coordinate of keypoints in the second image by the width of the first image
#         kp2.pt = (kp2.pt[0] + w1, kp2.pt[1])
#         color = tuple(np.random.randint(0, 255, 3).tolist())

#         # Draw a line between the matched keypoints
#         # cv2.line(vis, (int(kp1.pt[0]), int(kp1.pt[1])), (int(kp2.pt[0]), int(kp2.pt[1])), color, 1)

#     return vis


# def match_keypoints(keypoints1, keypoints2, max_distance = 3):
#     m = []
#     a = set()
#     b = set()

#     softmaxArrayDictA=[]
#     softmaxArrayDictB=[]
#     p = {}

#     # Use the keypoints1 directly without reshaping
#     treeA = KDTree(keypoints1, leaf_size=2)   
#     treeB = KDTree(keypoints2, leaf_size=2)   

#     for j, x in enumerate(keypoints2):
#         # Reshape x to a 2D array with the same dimensionality as keypoints1
#         x_reshaped = np.array(x).reshape(1, -1)
#         dist, i = treeA.query(x_reshaped, k=5)
#         # if len(i[0]) < 5:
#         #     return dataset[random.randint(0, 99)]
#         # Initialize an empty dictionary
#         result_dict = {}

#         #Iterate through the indices and distances
#         for q in range(len(dist[0])): 
#             if dist[0][q] <= max_distance: 
#                 # Add each index-distance pair to the dictionary
#                 result_dict[i[0][q]] = -dist[0][q]
                
        
#         # Apply softmax to the values if the dictionary is not empty
#         if result_dict:
#             softmax_values = softmax(list(result_dict.values()))
#             softmax_dict = {key: value for key, value in zip(result_dict.keys(), softmax_values)}
#             softmaxArrayDictB.append(softmax_dict)
#         else:
#             softmaxArrayDictB.append({})
        
        
#     for j, x in enumerate(keypoints1):
#         # Reshape x to a 2D array with the same dimensionality as keypoints1
#         x_reshaped = np.array(x).reshape(1, -1)
#         dist, i = treeB.query(x_reshaped, k=5)
        
#         # Initialize an empty dictionary
#         result_dict = {}

#         for q in range(5):
#             if dist[0][q] <= max_distance:
#                 # Add each index-distance pair to the dictionary
#                 result_dict[i[0][q]] = -dist[0][q]
        
    
#         # Apply softmax to the values if the dictionary is not empty
#         if result_dict:
#             softmax_values = softmax(list(result_dict.values()))
#             softmax_dict = {key: value for key, value in zip(result_dict.keys(), softmax_values)}
#             softmaxArrayDictA.append(softmax_dict)
#         else:
#             softmaxArrayDictA.append({})
        
        
#     #choose the high value in p
#     for i in range (len(softmaxArrayDictA)):
#         for j in range (len(softmaxArrayDictB)):
#             if softmaxArrayDictB[j].get(i) and softmaxArrayDictA[i].get(j):
#                 p[(i,j)] = softmaxArrayDictB[j].get(i)*softmaxArrayDictA[i].get(j)
#     sorted_p = dict(sorted(p.items(), key=lambda item: item[1]))

    
#     for key, value in sorted_p.items():
#         i = key[0] 
#         j = key[1]  
#         if value >= 0.25 and i not in a and j not in b:
#             m.append((i,j))
#             a.add(i)
#             b.add(j)
    
#     notA = set(range(len(keypoints1))) - a
#     notB = set(range(len(keypoints2))) - b

#     return m, notA, notB


# def match(images, matrix):
#     start=time.time()

#     img0, key1, des1= detect_and_draw_keypoints(images[0])
#     img1, key2, des2= detect_and_draw_keypoints(images[1])
#     if len(key1)<5 or len(key2)<5 :
#         return 0
#     # ig, axs = plt.subplots(1, 2, figsize=(10, 5))
#     # axs[0].imshow(images[0])
#     # axs[0].set_title('First Image')
#     # axs[0].axis('off')  # Hide axes ticks
#     # axs[1].imshow(images[1])
#     # axs[1].set_title('Second Image')
#     # axs[1].axis('off')  # Hide axes ticks
#     # plt.tight_layout()
#     # plt.show()

#     keypoint_coordinates1 = np.array([kp.pt for kp in key1])
#     keypoint_coordinates2 = np.array([kp.pt for kp in key2])

#     # Convert keypoints of the first image to homogeneous coordinates
#     homogeneous_coordinates1 = np.hstack((keypoint_coordinates1, np.ones((keypoint_coordinates1.shape[0], 1))))

#     # Apply the transformation matrix to the homogeneous coordinates of the first image
#     transformed_coordinates1 = np.dot(matrix, homogeneous_coordinates1.T).T

#     # Convert back to Cartesian coordinates by dividing by the third element (homogeneous coordinate)
#     transformed_coordinates_xy1 = transformed_coordinates1[:, :2] / transformed_coordinates1[:, 2, None]
   
#     m, notA, notB = match_keypoints(transformed_coordinates_xy1.tolist(), keypoint_coordinates2.tolist())

#    # Draw matching lines on the images
#     matched_image = draw_matching_lines(img0, key1, img1, key2, m)

#     plt.figure(figsize=(10, 5))
#     plt.imshow(matched_image)
#     plt.axis('off')
#     plt.tight_layout()

#     plt.show()
    

#     end=time.time()
#     # print("match time is",end-start)
#     return m, notA,notB, key1, key2, des1, des2


# # with open('data/data_Abyssinian_9.json', "w") as file:

# #     images, matrix = load_images_and_matrix('Abyssinian_9.jpg')
# #     result=match(images,matrix)
# #     m, notA, notB, key1, key2, des1, des2 = result                    
# #     key1 = list([(kp.pt[0], kp.pt[1]) for kp in key1])
# #     key2 = list([(kp.pt[0], kp.pt[1]) for kp in key2])

# #     des1 = list([row.tolist() for row in des1])
# #     des2 = list([row.tolist() for row in des2])

# #     notA = list(notA)
# #     notB = list(notB)

# #     m = list([(i[0],i[1]) for i in m])

# #     dic=({'key1': key1, 'des1': des1, 'key2': key2, 'des2': des2, 'notA': notA, 'notB': notB, 'm': m})
# #     # Save dictionary to JSON file 

# #     json.dump(dic, file)
# #     file.write('\n')







# # רשימה לאחסון זוגות האינדקסים
#         #     index_pairs = []

#         #     # מעבר על כל האינדקסים במטריצה
#         #     for i in range(P.shape[0]):  # עבור כל שורה
#         #         for j in range(P.shape[1]):  # עבור כל עמודה
#         #             if P[i, j] > 0.25:  # אם הערך גדול מ-0.25
#         #                 index_pairs.append([i, j])

#         # # המרת המערכים לסטים (כדי לבצע חיתוך)
#         #     set_1 = set(map(tuple, index_pairs))  # המרת כל זוג לרשימה על מנת לעבוד עם set
#         #     set_2 = set(map(tuple, data.get('m')))

#         # # חיתוך בין הסטים
#         #     intersection = set_1.intersection(set_2)

#         # # הצגת התוצאה
#         #     print(list(intersection))   



#          # index_pairs = torch.nonzero(P > 0.1).tolist()  # שליפת כל הזוגות עם דמיון מעל הסף
#             # matched_pairs = set(map(tuple, index_pairs))
#             # ground_truth_pairs = set(map(tuple, data.get('m')))
#             # intersection = matched_pairs.intersection(ground_truth_pairs)
#             # print(f"מספר התאמות נכונות: {len(intersection)} מתוך {len(ground_truth_pairs)}")

