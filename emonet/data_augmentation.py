# coding: utf8
import numpy as np
import cv2
from random import randint


def get_scale_center(bb):
    
    center = np.array([bb[2] - (bb[2]-bb[0])/2, bb[3] - (bb[3]-bb[1])/2])
    scale = (bb[2]-bb[0] + bb[3]-bb[1])/220.0

    return scale, center

def inv_mat(mat):
    ans = np.linalg.pinv(np.array(mat).tolist() + [[0,0,1]])
    return ans[:2]

def get_transform(center, scale, res, rot=0):
    # Generate transformation matrix
    
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1

    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 200
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))

    return t

class DataAugmentor(object):

    def __init__(self, target_width=256, target_height=256, random_translation = 0,
                 random_rotation=0, random_scaling=0, mirror=False, random_seed=None, shape_mirror_indx=None, flipping_probability=0.5):

        self.target_width = target_width
        self.target_height = target_height
        self.random_rotation = random_rotation
        self.random_scaling = random_scaling
        self.random_translation = random_translation
        self.mirror = mirror
        self.flipping_probability = flipping_probability

        if shape_mirror_indx is None:
            self.shape_mirror_indx = []
        else:
            self.shape_mirror_indx = shape_mirror_indx

        self.rng = np.random.RandomState(seed=random_seed)

    def __call__(self, image, bb=None, shape=None):
        """
            If bounding box is None, it assumes that the image is square and already cropped.
            The center will be the center of the image.
            Good for AffectNet.
        """
        #Checks that image is correct
        assert(image.ndim==3 and image.shape[2]==3)
        assert(image.dtype == np.uint8)

        if(bb is None):
            #Resize the image and the shape
            scalingFactor = self.target_height/image.shape[0] #Image is square
            image = cv2.resize(image, (self.target_width, self.target_height))

            #Resize the shape
            shape *= scalingFactor

            #######################################
            #Apply data augmentation : random scaling and rotation
            #######################################

            #Center is the middle of the image
            center = np.array([image.shape[1]/2, image.shape[0]/2])

            aug_rot = (self.rng.rand() * 2 - 1) * self.random_rotation # in deg.
            
            #Rotation and random scaling
            scale = self.rng.rand() * self.random_scaling*2 + (1-self.random_scaling) # ex: random_scaling is .25
            mat = cv2.getRotationMatrix2D((center[0], center[1]),aug_rot, scale)
            image = cv2.warpAffine(image, mat, (self.target_width, self.target_height))
            
            #Transforms the shape as well using homogeneous coordinates
            if shape is not None:
                shape = np.dot(np.concatenate((shape, shape[:, 0:1]*0+1), axis=1), mat.T)
            
            if self.random_translation!=0:
                dx = self.rng.randint(-self.random_translation * scale, self.random_translation * scale) # in px
                dy = self.rng.randint(-self.random_translation * scale, self.random_translation * scale)
            else:
                dx, dy = 0, 0
            
            #Translation
            mat = np.float32([[1,0, dx],[0,1, dy]])
            image = cv2.warpAffine(image, mat, (self.target_width, self.target_height))
            
            #Transforms the shape as well using homogeneous coordinates
            if shape is not None:
                shape = np.dot(np.concatenate((shape, shape[:, 0:1]*0+1), axis=1), mat.T)

            # Flip
            if np.random.randint(round(1.0/self.flipping_probability)) == 0 and self.mirror:
                image = image[:, ::-1]
                if shape is not None and shape.shape[0]==68:
                    shape = shape[self.shape_mirror_indx, :]

                shape[:, 0] = self.target_width - shape[:, 0]


        else:
            scale, center = get_scale_center(bb)

            aug_rot = (self.rng.rand() * 2 - 1) * self.random_rotation # in deg.
            aug_scale = self.rng.rand() * self.random_scaling*2 + (1-self.random_scaling) # ex: random_scaling is .25
            scale *= aug_scale

            if self.random_translation!=0:
                dx = self.rng.randint(-self.random_translation * scale, self.random_translation * scale)/center[0] # in px
                dy = self.rng.randint(-self.random_translation * scale, self.random_translation * scale)/center[1]
            else:
                dx, dy = 0, 0
            center[0] += dx * center[0]
            center[1] += dy * center[1]

            mat = get_transform(center, scale, (self.target_width, self.target_height), aug_rot)[:2]
            image = cv2.warpAffine(image, mat, (self.target_width, self.target_height))#, borderMode= cv2.BORDER_WRAP)

            if shape is not None:
                mat_pts = get_transform(center, scale, (self.target_width, self.target_height), aug_rot)[:2]
                shape = np.dot(np.concatenate((shape, shape[:, 0:1]*0+1), axis=1), mat_pts.T)

            # Flip
            if np.random.randint(round(1.0/self.flipping_probability)) == 0 and self.mirror:
                image = image[:, ::-1]
                if shape is not None and shape.shape[0]==68:
                    shape = shape[self.shape_mirror_indx, :]

                    shape[:, 0] = self.target_width - shape[:, 0]

        return image, shape
