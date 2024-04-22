import numpy as np
import torch


# TODO implement proper data loader for other data
class Preprocessor:

    def __init__(self, images=None, poses=None, focal=None, H=None, W=None):
        """
        Initialize the preprocessor with or without data
        If data is provided, it will be stored in the object, otherwise it will be loaded from a path
        :param images: images
        :param poses: poses
        :param focal: focal length
        :param H: height of the images
        :param W: width of the images
        """
        self.images = images
        self.poses = poses
        self.focal = focal
        self.H = H
        self.W = W

    def load_new_data(self, path):
        """
        Load data from a path
        :param path: path to the data
        :return: None
        """
        data = np.load(path)
        self.images = data['images']
        self.poses = data['poses']
        self.focal = data['focal']
        self.H, self.W = self.images.shape[1:3]

    def preprocess(self):
        # TODO if required, might not be necessary based on the data
        pass

    def split_data(self, split_params, randomize=False):
        """
        Split the data into training and testing sets
        :param split: split parameter in the form of (bool split_by_number, int number or float split_ratio) where
        split_by_number is a boolean indicating whether to split by number of samples or by ratio
        :param randomize: whether to randomize the data before splitting
        :return: training and testing images and poses as torch tensors
        """
        N = self.images.shape[0]
        split = split_params[1]
        if randomize:
            idx = torch.randperm(N)
        else:
            idx = torch.arange(N)

        if split_params[0]:
            train_idx = idx[:split]
            test_idx = idx[split:]
        else:
            train_idx = idx[:int(split*N)]
            test_idx = idx[int(split*N):]

        train_images = torch.tensor(self.images[train_idx])
        train_poses = torch.tensor(self.poses[train_idx])
        test_images = torch.tensor(self.images[test_idx])
        test_poses = torch.tensor(self.poses[test_idx])

        return train_images, train_poses, test_images, test_poses
