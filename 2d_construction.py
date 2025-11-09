import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def operation_on_depth_image(depth_image_path):
    img = cv2.imread(depth_image_path, cv2.IMREAD_ANYDEPTH)
    if img is None:
        print("Failed to load:", depth_image_path)
        return None
    image_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    depth_image_normalized = np.uint8(image_normalized)
    
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    depth_clahe = clahe.apply(depth_image_normalized)

    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    sharpened = cv2.filter2D(depth_clahe, -1, kernel)
    edges = cv2.Canny(sharpened, 50, 150)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped = sharpened[y-3:y+h+3, x-3:x+w+3]
    resized = cv2.resize(cropped, (128, 128))

    return resized

normal_object = ['boat']
uxo_object = ['morter_shell']

terrain_types = ['straightest_terrain', 'more_ruggedy_terrain','no_terrain','straigher_terrain']
const_save_path = f"E:\\He_is_enough03 X UniqoXTech X Dreams\\Click_here\\Artificial Intelligence\\CNN-SonarCloud\\2D_dataset"

for obj in normal_object:
    counting = 0
    for terrain in terrain_types:
        if terrain == 'no_terrain':
             mt_range = 1
        else:
             mt_range = 3     
        print(f"Terrain {terrain}")     
        for moved_terrain in range (0,3):
                print(f"moved {moved_terrain}")
                for orientation in range (1,9):
                    print(f"ori {orientation}")
                    for depth in range (0,8):
                        print(f"depth {depth}")
                        depth_image_path = f"E:\\He_is_enough03 X UniqoXTech X Dreams\\Click_here\\Artificial Intelligence\\CNN-SonarCloud\\boat_data\\{terrain}\\moved_terrain{moved_terrain}\\orientation_{orientation}\\depth\\depth_{depth}.tiff"
                        output = operation_on_depth_image(depth_image_path)
                        save_path = f"{const_save_path}\\{obj}"
                        os.makedirs(save_path, exist_ok=True)
                        output_path = os.path.join(save_path, f"{counting}.jpg")
                        counting = counting + 1
                        cv2.imwrite(output_path, output)



# output = operation_on_depth_image("E:\\He_is_enough03 X UniqoXTech X Dreams\\Click_here\\Artificial Intelligence\\CNN-SonarCloud\\boat_data\\straightest_terrain\\moved_terrain1\\orientation_3\\depth\\depth_3.tiff")
# cv2.imshow("Cropped Depth Image", output)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


