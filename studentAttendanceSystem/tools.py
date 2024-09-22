import cv2
import numpy as np
import glob
import os
import tqdm

class utilities():
    def __init__(self):
        pass
    def resize_with_padding(self,image, target_size):
        """
        Resizes an image to the target size while maintaining the aspect ratio and adding padding.

        Args:
            image (numpy.ndarray): Input image to be resized.
            target_size (tuple): Desired output size as (width, height).

        Returns:
            numpy.ndarray: Resized and padded image.
        """
        original_height, original_width = image.shape[:2] 
        # print(original_height,original_width)
        target_width, target_height = target_size 

        # Calculate the scaling factor
        scale = min(target_width / original_width, target_height / original_height)

        # Calculate the new dimensions
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        # Resize the image
        # resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

        # Create a new image with the target size and fill it with black pixels
        # padded_image = np.full((target_height, target_width, 3), 1, dtype=np.uint8)
        padded_image = np.full((target_height, target_width, 3), [255,255,255], dtype=np.uint8)
        # Calculate the top-left corner of the resized image in the padded image
        top_left_x = (target_width - new_width) // 2
        top_left_y = (target_height - new_height) // 2

        # Place the resized image onto the padded image
        padded_image[top_left_y:top_left_y + new_height, top_left_x:top_left_x + new_width] = resized_image
        return padded_image

    def main(self,path):
        # path = ".\Face_images"
        for img_path in tqdm.tqdm(glob.glob(os.path.join(path, "*\*.jpg")),desc = "Resizing in progress...."):
            path_list = img_path.split("\\")
            img = cv2.imread(img_path)

            directory_path = f".\\pre_processed_data\\{path_list[-2]}"
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
            target_size = (416, 416)  # Example target size
            resized_image = self.resize_with_padding(img, target_size)
            resized_image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"{directory_path}\{path_list[-1]}", resized_image)
            

        print("resizing done")


if __name__=="__main__":
    path = "mini_project\\face_images"
    obj = utilities()
    obj.main(path)