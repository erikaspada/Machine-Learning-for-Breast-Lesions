import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def creation_mask(image_path, save_mask_dir, save_breast_dir, show_results=False):
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian blur and thresholding to create a binary image
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv2.threshold(img_blur, 20, 255, cv2.THRESH_BINARY)

    # Find external contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)

    # Create the original mask (breast + background)
    mask_original = np.zeros_like(img)
    cv2.drawContours(mask_original, [max_contour], -1, color=255, thickness=cv2.FILLED)

    # Morphological closing to fill internal holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (60, 60))
    mask_cleaned = cv2.morphologyEx(mask_original, cv2.MORPH_CLOSE, kernel)

    # Erode to isolate only the outer curved edge
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (70, 70))
    mask_eroded = cv2.erode(mask_cleaned, kernel_erode, iterations=1)

    # Extract the border by subtracting the eroded mask from the cleaned mask
    mask_edge = cv2.subtract(mask_cleaned, mask_eroded)

    # Shrink the mask by removing the edge
    mask_shrunk = cv2.subtract(mask_cleaned, mask_edge)

    # Apply the shrunk mask to the original image
    result_shrunk_mask = cv2.bitwise_and(img, img, mask=mask_shrunk)

    # Show intermediate steps (optional)
    if show_results:
        fig, axs = plt.subplots(1, 4, figsize=(16, 5))

        axs[0].imshow(img, cmap='gray')
        axs[0].set_title("Original Image")
        axs[0].axis('off')

        axs[1].imshow(mask_cleaned, cmap='gray')
        axs[1].set_title("Cleaned Mask")
        axs[1].axis('off')

        axs[2].imshow(mask_shrunk, cmap='gray')
        axs[2].set_title("Shrunk Mask")
        axs[2].axis('off')

        axs[3].imshow(result_shrunk_mask, cmap='gray')
        axs[3].set_title("Masked Image")
        axs[3].axis('off')

        plt.tight_layout()
        plt.show()

    # Save the shrunk mask and the masked image
    base_name = os.path.basename(image_path).split(".")[0]
    cv2.imwrite(os.path.join(save_mask_dir, f"{base_name}_mask.jpeg"), mask_shrunk)
    cv2.imwrite(os.path.join(save_breast_dir, f"{base_name}_breast.jpeg"), result_shrunk_mask)

