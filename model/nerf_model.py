import torch


# TODO possibly fix issues and look into adding other features
class NRFModel(torch.nn.Module):
    def __init__(
            self,
            n_posenc_xyz=6,
            # n_posenc_dir=4,
            output_ch=4,
            n_hidden_xyz=256,
            # n_hidden_dir=128,
            n_layers_xyz=8,
            skips={4},
            batch_chunk=1024*32
    ):
        super().__init__()

        self.posenc_xyz = 3 + 3 * 2 * n_posenc_xyz
        self.n_posenc_xyz = n_posenc_xyz
        # self.posenc_dir = 3 + 3 * 2 * n_posenc_dir
        # self.n_posenc_dir = n_posenc_dir
        self.output_ch = output_ch
        self.hidden_xyz = n_hidden_xyz
        # self.hidden_dir = n_hidden_dir
        self.layers_xyz = n_layers_xyz
        # self.layers_dir = n_layers_dir
        self.skips = skips
        self.batch_chunk = batch_chunk

        self.xyz_encoding_fn = self.get_encoding_xyz_fn()

        def create_dense(in_dim, out_dim=self.hidden_xyz, act=torch.nn.ReLU(True)):
            linear_layer = torch.nn.Linear(in_dim, out_dim)
            torch.nn.init.xavier_uniform_(linear_layer.weight)
            return torch.nn.Sequential(
                linear_layer,
                act,
            )

        layers = [create_dense(self.posenc_xyz)]
        for i in range(1, self.layers_xyz):
            if i in self.skips:
                in_dim = self.hidden_xyz + self.posenc_xyz
            else:
                in_dim = self.hidden_xyz
            layer = create_dense(in_dim)
            layers.append(layer)

        in_dim = self.hidden_xyz
        out_dim = self.output_ch
        layers.append(torch.nn.Linear(in_dim, out_dim))

        self.mlp_xyz = torch.nn.ModuleList(layers)

    def forward(self, x):
        """
        Forward pass
        :param x: points sampled from the ray
        :return: RGB and density values
        """
        # flatten the points
        in_dims = x.shape
        x = x.reshape(-1, 3)
        results = []
        for i in range(0, x.shape[0], self.batch_chunk):
            curr_x = x[i:i+self.batch_chunk]
            curr_x = self.xyz_encoding_fn(curr_x)
            out = x[i:i+self.batch_chunk]
            out = self.xyz_encoding_fn(out)
            for j, layer in enumerate(self.mlp_xyz):
                if j in self.skips:
                    out = torch.cat([out, curr_x], dim=-1)
                out = layer(out)
            results.append(out)

        out = torch.cat(results, dim=0)
        out = out.reshape(list(in_dims[:-1]) + [self.output_ch])

        rgb = torch.sigmoid(out[..., :3])
        density = torch.nn.functional.relu(out[..., 3])

        return rgb, density

    def pos_enc_xyz(self, x):
        out = [x]
        funcs = [torch.sin, torch.cos]
        for i in range(self.n_posenc_xyz):
            for func in funcs:
                out.append(func(2 ** i * x))

        return torch.cat(out, dim=-1)

    def get_encoding_xyz_fn(self):
        if self.posenc_xyz == 0:
            return torch.nn.Identity
        else:
            return self.pos_enc_xyz
