import os
import requests
import zipfile
import tarfile
import cv2


# #download dataset Oxford Buildings
# url = "https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images.tgz"
# response = requests.get(url)

# # save
# with open('oxbuild_images.tgz', 'wb') as file:
#     file.write(response.content)

# # extract files
# with tarfile.open('oxbuild_images.tgz', 'r:gz') as tar_ref:
#     tar_ref.extractall('./landspace_images')
# with zipfile.ZipFile('lhq-1024.zip', 'r') as zip_ref:
#     zip_ref.extractall('./landspace_images')

# print("Oxford Buildings images downloaded and extracted.")


# resize and remove
def resize_images_in_folder(folder_path, target_size=(320, 240)):
    # Create a folder to save resized images
    resized_folder = os.path.join(folder_path, "resized")
    os.makedirs(resized_folder, exist_ok=True)

    for filename in os.listdir(folder_path):
        # Get the full path of the image
        file_path = os.path.join(folder_path, filename)

        # Skip directories
        if os.path.isdir(file_path):
            continue
        
        try:
            # Attempt to read the image
            img = cv2.imread(file_path)
            if img is None:
                raise ValueError("Image is None")

            # Resize the image
            resized_img = cv2.resize(img, target_size)

            # Save the resized image
            cv2.imwrite(os.path.join(resized_folder, filename), resized_img)

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            # If there's an error, delete the corrupted image
            os.remove(file_path)

# Usage
folder_path = 'landspace_images/dataset'  
resize_images_in_folder(folder_path)

