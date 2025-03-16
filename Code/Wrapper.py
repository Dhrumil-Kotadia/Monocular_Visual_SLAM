import cv2
import json
import scipy
import dbow
import torch

import numpy as np

from glob import glob
from ultralytics import YOLO
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares


def Load_Images(Path):
    Images = []
    for file in sorted(glob(Path + '*.png')):
        img = cv2.imread(file)
        Images.append(img)
    return Images

class visual_slam:
    """
    Class to perform Visual SLAM using ORB features and Bundle Adjustment.
    """
    def __init__(self, k_matrix):
        """
        Initialize the Visual SLAM object.
        """
        self.k_matrix = k_matrix
        self.detector = cv2.ORB_create(500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        self.global_rotation = np.eye(3)
        self.global_translation = np.zeros((3, 1))
        self.global_transfomation = np.eye(4)
        
        self.rotations = []
        self.images = []
        self.descriptors = []
        self.keypoints = []
        
        self.remove_dynamic = True
        self.loop_closure = False
        
        self.poses = []
        self.poses_ba = []
        self.final_points_3d = []
        self.final_points_3d_ba = []
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO('yolov8n.pt').to(device)
        self.yolo_confidence = 0.3
        
        self.bows = []
        self.dbow_vocab = dbow.Vocabulary.load("vocab.pkl")
        
        
    def Extract_Features(self, img):
        """
        Extract ORB features from the input image.

        Args:
            img: Input image.
        """
        keypoints, descriptors = self.detector.detectAndCompute(img, None)
        keypoints, descriptors = self.remove_keypoints(list(keypoints), list(descriptors))
        
        return keypoints, descriptors

    def remove_keypoints(self, keypoints, descriptors):
        """
        Remove keypoints detected by YOLO from the list of keypoints.

        Args:
            keypoints: List of keypoints detected by ORB.
            descriptors: List of descriptors corresponding to the keypoints.
        """
        results = self.model(self.images[-1], stream=True, conf=self.yolo_confidence, verbose=False)
        for result in results:
            boxes = result.boxes.xyxy.to('cpu').numpy().astype(int)
            classes = result.boxes.cls.to('cpu').numpy().astype(int)
        indices = []
        for box in boxes:
            for kp, des in zip(keypoints, descriptors):
                if box[0] < kp.pt[0] < box[2] and box[1] < kp.pt[1] < box[3] and classes[0] in [0, 2]:
                    indices.append(keypoints.index(kp))
        indices = list(set(indices))
        for index in sorted(indices, reverse=True):
            del keypoints[index]
            del descriptors[index]

        return tuple(keypoints), np.array(descriptors) if len(descriptors) > 0 else np.array([])

    def Match_Features(self, Descriptors1, Descriptors2):
        """
        Match ORB features between two images.

        Args:
            Descriptors1: Descriptors of the first image.
            Descriptors2: Descriptors of the second image.
        """
        matches = self.matcher.match(Descriptors1, Descriptors2)
        matches = sorted(matches, key = lambda x:x.distance)

        return matches

    def estimate_pose(self, kp1, kp2, matches, K):
        """
        Estimate the pose of the camera using the matched keypoints.

        Args:
            kp1: Keypoints of the first image.
            kp2: Keypoints of the second image.
            matches: Matched keypoints between the two images.
            K: Camera matrix.
        """
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC)
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K, mask=mask)

        return R, t, mask

    def ExtractCameraPose(E):
        """
        Extract camera pose from the essential matrix.

        Args:
            E: Essential matrix.
        """
        W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        U, S, Vt = scipy.linalg.svd(E)
        R1 = np.dot(np.dot(U, W), Vt)
        R2 = np.dot(np.dot(U, W.T), Vt)
        T1 = U[:, 2]
        T2 = -U[:, 2]
        R = [R1, R1, R2, R2]
        T = [T1, T2, T1, T2]
        for i in range(4):
            if np.linalg.det(R[i]) < 0:
                R[i] = -R[i]
                T[i] = -T[i]

        return R, T

    def triangulate_points(self, kp1, kp2, matches, R, T, r, t, K, mask):
        """
        Triangulate 3D points from the matched keypoints.

        Args:
            kp1: Keypoints of the first image.
            kp2: Keypoints of the second image.
            matches: Matched keypoints between the two images.
            R: Rotation matrix.
            T: Translation vector.
            r: Rotation matrix.
            t: Translation vector.
            K: Camera matrix.
            mask: Mask of the matched keypoints.
        """
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])[mask.ravel() == 1]
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])[mask.ravel() == 1]
        
        if pts1.shape[0] < 5 or pts2.shape[0] < 5:
            return [], [], []
        
        u_pts1 = cv2.undistortPoints(np.expand_dims(pts1, axis=1), K, None)
        u_pts2 = cv2.undistortPoints(np.expand_dims(pts2, axis=1), K, None)
        
        P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = np.hstack((r, t))
        
        points_4d = cv2.triangulatePoints(P1, P2, u_pts1, u_pts2)
        points_3d = points_4d[:3] / points_4d[3]

        T_Total = T + np.dot(R, t)
        R_Total = np.dot(r, R)
        points_3d_World = np.dot(R_Total.T, points_3d) + np.tile(T_Total, (points_3d.shape[1]))

        return points_3d_World.T, pts1, pts2

    def check_loop_closure(self):
        """
        Check for loop closure in the trajectory.
        """
        if len(self.descriptors) < 2:
            return 0
        
        if len(self.descriptors) < 3:
            descriptors = [dbow.ORB.from_cv_descriptor(desc) for desc in self.descriptors[-2]]
            self.bows.append(self.dbow_vocab.descs_to_bow(descriptors))
            descriptors = [dbow.ORB.from_cv_descriptor(desc) for desc in self.descriptors[-1]]
            self.bows.append(self.dbow_vocab.descs_to_bow(descriptors))
            return 0
        
        descriptors = [dbow.ORB.from_cv_descriptor(desc) for desc in self.descriptors[-1]]
        bow = self.dbow_vocab.descs_to_bow(descriptors)
        distances = []

        for j in range(len(self.bows)):
            distances.append(bow.score(self.bows[j]))
        
        if len(self.images) > 50:
            distances = np.array(distances)
            valid_indices = np.where((distances > 0.53) & (np.arange(len(distances)) <= len(self.bows) - 200))[0]

            if valid_indices.size > 0:
                valid_index = valid_indices[np.argmax(distances[valid_indices])]  # Get index of max valid distance
                print("Loop Closure Detected!")
                print("Valid Index:", valid_index)
                self.global_rotation = self.rotations[valid_index]
                self.global_translation = self.poses[valid_index]
                self.poses = self.poses[valid_index]
                self.rotations = self.rotations[valid_index]    
                self.loop_closure = True
                self.Bundle_Adjustment()

        self.bows.append(bow)

    def reprojection_error(self, params, n_cameras, n_points, camera_indices, point_indices, points_2d, k_matrix):
        """
        Compute reprojection error given the camera parameters and 3D points.
        """
        camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
        points_3d = params[n_cameras * 6:].reshape((n_points, 3))
        
        projected_points = []

        for i in range(len(camera_indices)):
            cam_idx = camera_indices[i]
            pt_idx = point_indices[i]
            
            r_vec = camera_params[cam_idx, :3]
            t_vec = camera_params[cam_idx, 3:]
            
            R, _ = cv2.Rodrigues(r_vec)
            p_cam = R @ points_3d[pt_idx] + t_vec
            
            p_proj = k_matrix @ p_cam
            p_proj /= p_proj[2]
            
            projected_points.append(p_proj[:2])
        
        return (np.array(projected_points) - points_2d).ravel()

    def Bundle_Adjustment(self):
        """
        Perform bundle adjustment to refine camera poses and 3D points.
        """
        n_cameras = len(self.poses)
        n_points = len(self.final_points_3d)
        camera_params = np.zeros((n_cameras, 6))
        
        for i in range(n_cameras):
            R, _ = cv2.Rodrigues(np.array(self.rotations[i]))
            camera_params[i, :3] = R.ravel()
            camera_params[i, 3:] = [self.poses[i][0][0], self.poses[i][1][0], self.poses[i][2][0]]
        
        camera_params = camera_params.ravel()
        points_3d = np.array(self.final_points_3d).ravel()
        
        camera_indices = []
        point_indices = []
        points_2d = []
        
        for i, features in enumerate(self.keypoints):
            for j, feature in enumerate(features):
                camera_indices.append(i)
                point_indices.append(j)
                points_2d.append(list(feature))
        
        camera_indices = np.array(camera_indices)
        point_indices = np.array(point_indices)
        points_2d = np.array(points_2d)
        
        params = np.hstack((camera_params, points_3d))
        
        A = lil_matrix((len(points_2d) * 2, len(params)), dtype=int)
        
        for i in range(len(points_2d)):
            A[2 * i:2 * i + 2, camera_indices[i] * 6:camera_indices[i] * 6 + 6] = 1
            A[2 * i:2 * i + 2, n_cameras * 6 + point_indices[i] * 3:n_cameras * 6 + point_indices[i] * 3 + 3] = 1
        
        res = least_squares(
            self.reprojection_error, params, jac_sparsity=A, verbose=2, max_nfev=1000,
            args=(n_cameras, n_points, camera_indices, point_indices, points_2d, self.k_matrix)
        )
        
        refined_params = res.x
        self.update_bundle_adjustment_results(refined_params, n_cameras, n_points)

    def update_bundle_adjustment_results(self, refined_params, n_cameras, n_points):
        """
        Update camera poses and 3D points after bundle adjustment.
        """
        refined_camera_params = refined_params[:n_cameras * 6].reshape((n_cameras, 6))
        refined_points_3d = refined_params[n_cameras * 6:].reshape((n_points, 3))
        translations = []
        rotations = []

        for i in range(n_cameras):
            r_vec = refined_camera_params[i, :3]
            t_vec = refined_camera_params[i, 3:]
            
            R, _ = cv2.Rodrigues(r_vec)
            
            translations.append(t_vec)
            rotations.append(R)
        
        self.global_rotation = rotations[-1]
        self.global_translation = translations[-1]
        self.global_translation = np.expand_dims(translations[-1], axis=1)
        self.final_points_3d_ba = refined_points_3d.tolist()
        self.poses_ba = translations

    def run(self, image):
        """
        Run the Visual SLAM pipeline.

        Args:
            image: Input image.
        """
        self.images.append(image)
        if len(self.images) < 2:
            Features1, Descriptors1 = self.Extract_Features(self.images[-1])
            self.descriptors.append(Descriptors1)
            return 0
        
        print("Processing Frame: ", len(self.images))
        Features1, Descriptors1 = self.Extract_Features(self.images[-2])
        Features2, Descriptors2 = self.Extract_Features(self.images[-1])
        count = 0
        
        Matches = self.Match_Features(Descriptors1, Descriptors2)
        r, t, mask = self.estimate_pose(Features1, Features2, Matches, self.k_matrix)
        Points3D, Features1, Features2 = self.triangulate_points(Features1, Features2, Matches, self.global_rotation, self.global_translation, r, t, self.k_matrix, mask)
        
        if len(Points3D) == 0:
            self.images.pop()
            return 0
        
        for Iter in range(len(Points3D)):
            UpdatedPoints3D = Points3D[Iter]
            self.final_points_3d.append(UpdatedPoints3D)

        # # Check if features1 is present in self.keypoints
        # if len(self.keypoints) > 0:
        #     for kp1 in Features1:
        #         for kp2 in self.keypoints[-1]:
        #             if kp1.tolist() == kp2.tolist():
        #                 count += 1
        # print("Count: ", count)

        self.descriptors.append(Descriptors2)
        self.keypoints.append(Features2)
        self.check_loop_closure()
        if self.loop_closure is False:
            self.global_translation = self.global_translation + np.dot(self.global_rotation, t)
            self.global_rotation = np.dot(r, self.global_rotation)
            self.poses.append(self.global_translation.tolist())
            self.rotations.append(self.global_rotation.tolist())
        else:
            self.loop_closure = False
        

def main():
    
    Path = "../image_0/"
    print("Reading Images...")
    Images = Load_Images(Path)
    
    K = np.array([[707.0912, 0, 601.8873],
        [0, 707.0912, 183.1104],
        [0, 0, 1]])

    visual_slam_obj = visual_slam(K)
    for i in range(len(Images)):
        visual_slam_obj.run(Images[i])
    
    visual_slam_obj.Bundle_Adjustment()
    F = open("Poses_temp.json", "w")
    Dict = {}
    
    # Save the camera poses and 3D points to a JSON file

    for i in range(len(visual_slam_obj.poses)):
        Dict[i] = [visual_slam_obj.poses[i][0][0], visual_slam_obj.poses[i][1][0], visual_slam_obj.poses[i][2][0]]
    json.dump(Dict, indent=4, fp=F)
    F.close()
    F = open("Points3D_temp.json", "w")
    Dict = {}
    for i in range(len(visual_slam_obj.final_points_3d)):
        Dict[i] = visual_slam_obj.final_points_3d[i].tolist()
    json.dump(Dict, indent=4, fp=F)
    F.close()

    F = open("Poses_ba.json", "w")
    Dict = {}
    for i in range(len(visual_slam_obj.poses_ba)):
        Dict[i] = [visual_slam_obj.poses_ba[i][0], visual_slam_obj.poses_ba[i][1], visual_slam_obj.poses_ba[i][2]]
    json.dump(Dict, indent=4, fp=F)
    F.close()

    F = open("Points3D_ba.json", "w")
    Dict = {}
    for i in range(len(visual_slam_obj.final_points_3d_ba)):
        Dict[i] = [visual_slam_obj.final_points_3d_ba[i][0], visual_slam_obj.final_points_3d_ba[i][1], visual_slam_obj.final_points_3d_ba[i][2]]
    json.dump(Dict, indent=4, fp=F)
    F.close()

if __name__ == "__main__":
    main()

