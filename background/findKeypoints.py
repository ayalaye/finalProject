import cv2
import matplotlib.pyplot as plt

def detect_and_draw_keypoints(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Initialize the SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(image, None)
    keypoints = sorted(keypoints, key=lambda x: -x.response)
    keypoints = keypoints[:30]
    # Print the number of keypoints detected
    print(f"Number of keypoints: {len(keypoints)}")

    # Print the shape of the descriptors array
    print(f"Shape of descriptors: {descriptors.shape}")

    # Iterate through keypoints and print descriptors
    for i, keypoint in enumerate(keypoints):
        x, y = keypoint.pt
        print(f"Keypoint {i} - Coordinates: ({x}, {y}) - Descriptor: {descriptors[i]}")

    # Draw keypoints on the image
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)

    # Display the image with keypoints
    plt.imshow(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Turn off axis labels
    plt.show()

# Example usage:
image_path = r'C:\Users\asyer\Desktop\final\finalProject\others\images.jpg'
detect_and_draw_keypoints(image_path)
