import torch

from model.nerf_model import NRFModel


# TODO possibly fix issues and look into adding other features
class NRFRenderer(torch.nn.Module):
    def __init__(
            self,
            # for the renderer
            # TODO
            num_samples=64,
            near=2.0,
            far=6.0,
            random_samples=False,
            # for the model
            n_posenc_xyz=6,
            # n_posenc_dir=4,
            output_ch=4,
            n_hidden_xyz=256,
            # n_hidden_dir=128,
            n_layers_xyz=8,
            skips={4},
            batch_chunk=1024 * 32
    ):
        super().__init__()

        self.img_H = None
        self.img_W = None

        self.num_samples = num_samples
        self.near = near
        self.far = far
        self.random_samples = random_samples

        # Initialize the model to get rgb and density values from the points
        self.implicit_model = NRFModel(
            n_posenc_xyz=n_posenc_xyz,
            output_ch=output_ch,
            n_hidden_xyz=n_hidden_xyz,
            n_layers_xyz=n_layers_xyz,
            skips=skips,
            batch_chunk=batch_chunk
        )

        # TODO: Initialize the renderer and do volume rendering

        # pipeline for the renderer:
        # 1. take in rays or create rays from H, W, focal depending on how the renderer is initialized
        # 2. sample points along the rays
        # 3. get the rgb and density values from the model
        # 4. do volume rendering to get the final image

    def get_rays(self, H, W, focal, pose):
        '''
        Get rays from the poses and the camera parameters
        :param H: height of the image
        :param W: width of the image
        :param focal: focal length of the camera
        :param pose: pose of the camera
        :return: rays
        '''
        self.img_H = H
        self.img_W = W

        i, j = torch.meshgrid(torch.arange(float(W)), torch.arange(float(H)), indexing='xy')
        # get transformations from the pose
        rotation = pose[:3, :3]
        translation = pose[:3, 3]
        # get middle of each pixel for the ray directions
        directions = torch.stack([
            (i - W * 0.5) / focal, -(j - H * 0.5) / focal, -torch.ones_like(i)
        ], dim=-1).to(rotation.device)
        # get the ray directions
        ray_directions = torch.sum(directions[..., None, :] * rotation, dim=-1)
        # normalize the ray directions
        ray_directions /= torch.norm(ray_directions, dim=-1, keepdim=True)
        # get the ray origin
        ray_origins = translation.expand_as(ray_directions)

        return ray_origins, ray_directions

    def sample_points(self, ray_origins, ray_directions):
        """
        Sample points along the rays
        :param rays: rays
        :return: points and t values
        """
        t_vals = torch.linspace(self.near, self.far, self.num_samples, device=ray_origins.device)
        if self.random_samples:
            # TODO
            pass
        # add 3D dimension to the origins and directions and find sample points
        points = ray_origins[..., None, :] + t_vals[..., :, None] * ray_directions[..., None, :]

        return points, t_vals

    def forward(self, rays):
        """
        Forward pass
        :param rays: rays
        :return: rgb, depth
        """
        ray_origins, ray_directions = rays

        query_points, t_values = self.sample_points(ray_origins, ray_directions)

        rgb, density = self.implicit_model(query_points)

        pairwise_diff = t_values[..., 1:] - t_values[..., :-1]
        # initialize depth values, set the last value to a large number to indicate end of the ray
        dists = torch.cat([pairwise_diff, torch.tensor([1e10]).to(pairwise_diff)], dim=-1)
        # compute transmittance
        alpha = 1.0 - torch.exp(-density * dists)
        # compute opacity accumulated along the ray and scale by density contribution
        complement = 1.0 - alpha + 1e-10  # non-zero
        alpha_composite = torch.cumprod(complement, dim=-1) * alpha

        rgb_map = torch.sum(alpha_composite[..., None] * rgb, dim=-2)
        depth_map = torch.sum(alpha_composite * t_values, dim=-1)

        return rgb_map, depth_map
