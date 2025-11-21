import torch
from train import visualize_image, create_patches, create_polynomial_features, reconstruct_from_patches,apply_filters,binarize_objectness
from model import DoubleConvGaussian
import numpy as np
import os
from import_dataset import batched_import

import torch

class StreamingReconstructor:
    def __init__(self, C, H, W, patch_size=32, stride=None, device=None, dtype=None):
        self.C, self.H, self.W = C, H, W
        self.K = patch_size
        self.S = patch_size if stride is None else stride
        self.device = device
        self.dtype = dtype

        # Accumulator for pixel sums and weights
        self.sum_canvas = torch.zeros((C, H, W), device=device, dtype=dtype)
        self.weight_canvas = torch.zeros((1, H, W), device=device, dtype=dtype)

        # Precompute grid sizes
        self.nh = (H - self.K) // self.S + 1
        self.nw = (W - self.K) // self.S + 1
        self.total = self.nh * self.nw

    def _ij_from_index(self, p):
        i = p // self.nw
        j = p %  self.nw
        return i, j

    def add_patch(self, patch, p=None, ij=None):
        """
        patch: (C, K, K)
        p: optional flat index (if patches came from `create_patches` in raster order)
        ij: optional (i, j) grid coordinates if you know them directly
        Exactly one of (p, ij) must be provided.
        """
        if (p is None) == (ij is None):
            raise ValueError("Provide exactly one of p or ij")

        if ij is None:
            i, j = self._ij_from_index(p)
        else:
            i, j = ij

        y0, x0 = i * self.S, j * self.S
        y1, x1 = y0 + self.K, x0 + self.K

        self.sum_canvas[:, y0:y1, x0:x1] += patch
        self.weight_canvas[:, y0:y1, x0:x1] += 1.0

    def finalize(self):
        # Avoid divide-by-zero (outside covered region, if any)
        weight = torch.where(self.weight_canvas == 0,
                             torch.ones_like(self.weight_canvas),
                             self.weight_canvas)
        recon = self.sum_canvas / weight
        return recon



def test_loop(model,x_image,x_filters,y,patch_size=32,weights_path='weights/DoubleConvGaussian.pt'):
    x_image = torch.from_numpy(x_image).float()/255
    x_filters = torch.from_numpy(x_filters).float()/255
    x_stacked = torch.cat((x_image,x_filters),dim=-1)
    y = torch.from_numpy(y).float()
    reconstructors = [StreamingReconstructor(1,x_image.shape[1],x_image.shape[2]) for i in range(y.shape[1])]
    for i in range(x_image.shape[0]):
        # Permute the image from (H, W, C) to (C, H, W) before creating patches
        image_chw_pos = x_stacked[i].permute(2, 0, 1)
        image_chw_neg = 1 - image_chw_pos
        image_chw = torch.concatenate((image_chw_pos, image_chw_neg), dim=0)
        label_pos_chw = y[i][:,:,:,0]
        label_neg_chw = y[i][:,:,:,1]
        patched_image = create_patches(image_chw, patch_size)
        patched_pos_chw = create_patches(label_pos_chw, patch_size)
        patched_neg_chw = create_patches(label_neg_chw, patch_size)
        patched_pos_chw = torch.where(patched_pos_chw>0,patched_pos_chw,0)
        index_positive = torch.unique(torch.where(patched_pos_chw>0)[0])
        # patched_pos_chw = patched_pos_chw[index_positive]

        if index_positive.shape[0]>0:
            for j in index_positive:
                patch = patched_image[j]
                patch_pos = patched_pos_chw[j].float()
                polynomial_patch = create_polynomial_features(patch)
                pred = model(polynomial_patch).float()
                for k in range(y.shape[1]):
                    if pred[k].max().item()>0:
                        print(pred[k].max())
                        input('yipo')
                        reconstructors[k].add_patch(pred[k], j)
                    else:
                        print("patch max is ",patch_pos[k].max().item())
                        print("for class ",k,"pred is ",pred[k].max().item())
                print("patch j:",j)
            batch_pred = [r.finalize() for r in reconstructors]
            batch_pred = torch.stack(batch_pred, dim=0)
            print(batch_pred.shape)
            input('yipo')
            


def main():
    weights_dir = 'weights'
    weights_path = os.path.join(weights_dir, 'DoubleConvGaussian.pt')
    os.makedirs(weights_dir, exist_ok=True)
    model = DoubleConvGaussian(num_channels=4525, num_classes=12, kernel_size=3, num_sub_features = 256, num_bands=10)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path))
        print(f"Loaded model weights from {weights_path}")
    for i in range(100):
        batch_image, batch_label = batched_import(batch_size=2)
        if not batch_label:
            continue
        numpy_batch_image = np.array(batch_image)
        numpy_processed_batch = apply_filters(numpy_batch_image)
        processed_label_batch = binarize_objectness(batch_image,batch_label)
        test_loop(model,numpy_batch_image,numpy_processed_batch,processed_label_batch)

if __name__ == "__main__":
    main()