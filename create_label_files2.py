import numpy as np
from pykdtree.kdtree import KDTree
import trimesh
import cv2
import glob
import os
import sys
from tqdm import trange
from scipy.optimize import minimize
from config.registrationParameters import *
import json

def get_camera_intrinsic(folder):
    with open(folder + 'intrinsics.json', 'r') as f:
        camera_intrinsics = json.load(f)

    K = np.zeros((3, 3), dtype='float64')
    K[0, 0], K[0, 2] = float(camera_intrinsics['fx']), float(camera_intrinsics['ppx'])
    K[1, 1], K[1, 2] = float(camera_intrinsics['fy']), float(camera_intrinsics['ppy'])
    K[2, 2] = 1.
    return camera_intrinsics, K

# Function to compute 2D projections from 3D points using camera intrinsics
def compute_projection(points_3D, internal_calibration):
    points_3D = points_3D.T
    projections_2d = np.zeros((2, points_3D.shape[1]), dtype='float32')
    camera_projection = internal_calibration.dot(points_3D)
    projections_2d[0, :] = camera_projection[0, :] / camera_projection[2, :]
    projections_2d[1, :] = camera_projection[1, :] / camera_projection[2, :]
    return projections_2d

# Function to print usage instructions
def print_usage():
    print("Usage: create_label_files.py <path>")
    print("path: all or name of the folder")
    print("e.g., create_label_files.py all, create_label_files.py LINEMOD/Cheezit")

if __name__ == "__main__":
    try:
        if sys.argv[1] == "all":
            folders = glob.glob("LINEMOD/*/")
        elif sys.argv[1] + "/" in glob.glob("LINEMOD/*/"):
            folders = [sys.argv[1] + "/"]
        else:
            print_usage()
            sys.exit()
    except IndexError:
        print_usage()
        sys.exit()

    # Iterate through each folder
    for classlabel, folder in enumerate(folders):
        print("{} is assigned class label {}.".format(folder[8:-1], classlabel))
        camera_intrinsics, K = get_camera_intrinsic(folder)
        
        # Create directories if they do not exist
        path_label = folder + "labels"
        if not os.path.exists(path_label):
            os.makedirs(path_label)

        path_mask = folder + "mask"
        if not os.path.exists(path_mask):
            os.makedirs(path_mask)

        path_transforms = folder + "transforms"
        if not os.path.exists(path_transforms):
            os.makedirs(path_transforms)

        # Load transformation matrices
        transforms_file = folder + 'transforms.npy'
        try:
            transforms = np.load(transforms_file)
        except IOError:
            print("Transforms not computed, run compute_gt_poses.py first")
            continue
        
        # Load mesh and compute OBB
        mesh = trimesh.load(folder + "registeredScene.ply")
        Tform = mesh.apply_obb()
        mesh.export(file_obj=folder + folder[8:-1] + ".ply")

        # Compute bounding box dimensions
        points = mesh.bounding_box.vertices
        center = mesh.centroid
        min_x = np.min(points[:, 0])
        min_y = np.min(points[:, 1])
        min_z = np.min(points[:, 2])
        max_x = np.max(points[:, 0])
        max_y = np.max(points[:, 1])
        max_z = np.max(points[:, 2])
        
        # Define bounding box corners
        points = np.array([[min_x, min_y, min_z], [min_x, min_y, max_z], [min_x, max_y, min_z],
                           [min_x, max_y, max_z], [max_x, min_y, min_z], [max_x, min_y, max_z],
                           [max_x, max_y, min_z], [max_x, max_y, max_z]])

        print("{}\tmin_x={}\tmin_y={}\tmin_z={}\tsize_x={}\tsize_y={}\tsize_z={}".format(
            folder, min_x, min_y, min_z, (max_x - min_x), (max_y - min_y), (max_z - min_z)))

        # Define points for projections
        points_original = np.concatenate((np.array([[center[0], center[1], center[2]]]), points))
        points_original = trimesh.transformations.transform_points(points_original, np.linalg.inv(Tform))

        # Process each frame
        for i in trange(len(transforms)):
            mesh_copy = mesh.copy()
            img = cv2.imread(folder + "JPEGImages/" + str(i * LABEL_INTERVAL) + ".jpg")
            transform = np.linalg.inv(transforms[i])
            transformed = trimesh.transformations.transform_points(points_original, transform)

            corners = compute_projection(transformed, K)
            corners = corners.T
            corners[:, 0] = corners[:, 0] / int(camera_intrinsics['width'])
            corners[:, 1] = corners[:, 1] / int(camera_intrinsics['height'])

            T = transform  # T is already 3x4

            # Save the transformation matrix directly as 3x4
            filename = path_transforms + "/" + str(i * LABEL_INTERVAL) + ".npy"
            np.save(filename, T)
            
            sample_points = mesh_copy.sample(10000)
            masks = compute_projection(sample_points, K)
            masks = masks.T

            min_x = np.min(masks[:, 0])
            min_y = np.min(masks[:, 1])
            max_x = np.max(masks[:, 0])
            max_y = np.max(masks[:, 1])

            image_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            for pixel in masks:
                cv2.circle(image_mask, (int(round(pixel[0])), int(round(pixel[1]))), 5, 255, -1)
   
            thresh = cv2.threshold(image_mask, 30, 255, cv2.THRESH_BINARY)[1]
            
            # Handle the different return values of cv2.findContours in OpenCV 2.x and 3.x+
            contours_info = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours_info) == 3:
                _, contours, hierarchy = contours_info
            else:
                contours, hierarchy = contours_info

            cnt = max(contours, key=cv2.contourArea)
    
            image_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.drawContours(image_mask, [cnt], -1, 255, -1)

            mask_path = path_mask + "/" + str(i * LABEL_INTERVAL) + ".png"
            cv2.imwrite(mask_path, image_mask)
                
            with open(path_label + "/" + str(i * LABEL_INTERVAL) + ".txt", "w") as file:
                message = "{} ".format(str(classlabel)[:8])
                file.write(message)
                for pixel in corners:
                    for digit in pixel:
                        message = "{} ".format(str(digit)[:8])
                        file.write(message)
                message = "{} ".format(str((max_x - min_x) / float(camera_intrinsics['width']))[:8])
                file.write(message) 
                message = "{}".format(str((max_y - min_y) / float(camera_intrinsics['height']))[:8])
                file.write(message)

