from pre_process import apply_excess_blue, apply_excess_green, apply_grayness
from import_dataset import batched_import
from model import GMM
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

def apply_filters(image_batch):
    """
    Applies a series of filters to the image batch and stacks the results.
    Note: The number of filters applied is determined by the implementations
    of the filter functions, not the parameters of this function.
    """    
    # All outputs should be flat: a list of (H, W) arrays
    def flatten(batch):
        """Takes a batch of (H,W,C) or (H,W) and returns a list of (H,W) slices"""
        if isinstance(batch, list):
            result = []
            for item in batch:
                if item.ndim == 2:
                    result.append(item)
                elif item.ndim == 3:
                    # split along channel
                    result.extend([item[..., c] for c in range(item.shape[-1])])
                else:
                    raise ValueError("Unexpected shape in batch:", item.shape)
            return result
        else:
            raise ValueError("Input to flatten must be a list")

    all_processed = np.stack((
        np.stack(flatten(excess_blue_processed := apply_excess_blue(image_batch)) ,axis=0),
        np.stack(flatten(excess_green_processed := apply_excess_green(image_batch)) ,axis=0),
        np.stack(flatten(grayness_processed := apply_grayness(image_batch)) ,axis=0),
    ),axis=3)
    return all_processed


def pre_center_centroids(model, weights_path):
    mu_3 = torch.stack([model.gaussians[i].mu.data for i in range(3)]).requires_grad_(True)
    if model.gaussians[4].mu.data[0].item()==0.5:
        for i in range(100):
            mu_4 = model.gaussians[3].mu.data
            mu_4 = mu_4.requires_grad_(True)
            mu_3_extra = torch.concatenate((mu_3,mu_4.unsqueeze(0)),dim=0)
            mu_5 = model.gaussians[4].mu.data
            mu_5 = mu_5.requires_grad_(True)
            grad_4 = torch.autograd.grad(-((mu_4-mu_3)**2).mean(),mu_4)[0]
            grad_5 = torch.autograd.grad(-((mu_5-mu_4)**2).mean(),mu_5)[0]
            with torch.no_grad():
                model.gaussians[3].mu.add_(grad_4, alpha=-0.01)
                model.gaussians[3].mu.clamp_(-1,1)
                model.gaussians[4].mu.add_(grad_5, alpha=-0.01)
                model.gaussians[4].mu.clamp_(-1,1)
            torch.save(model.state_dict(), weights_path)
    return model


def kl_to_identity_from_L(L):
    # L is lower-triangular with positive diagonal (use softplus+eps when building it)
    tr = (L**2).sum()                            # tr(LL^T) = ||L||_F^2
    logdet = 2.0 * torch.log(torch.diagonal(L, 0)).sum()
    d = L.size(0)
    return tr - logdet - d

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights_dir = 'weights'
    weights_path = os.path.join(weights_dir, 'GMM.pt')
    os.makedirs(weights_dir, exist_ok=True)
    model = GMM(in_channels=3,num_gaussians=5).to(device)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path))
        print(f"Loaded model weights from {weights_path}")
        model = pre_center_centroids(model, weights_path)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00001)
        lambda_weight = 1e-2
        for k in range(2):
            batch_image, _ = batched_import(batch_size=8)
            batch_image = (torch.from_numpy(np.array(batch_image)).reshape(8*512*512,3)/255).to(device)
            for i in range(8*512*512//2000):
                optimizer.zero_grad()
                batch = batch_image[i*2000:(i+1)*2000]
                pred = model.forward_loss(batch)
                loss = pred.mean()
                log_pi = torch.log_softmax(model.pi,dim=0)
                pi = torch.exp(log_pi)
                entropy_penalty = lambda_weight*torch.sum(pi*(log_pi-torch.log(torch.tensor(1/5).to(device))))
                loss += entropy_penalty
                for i in range(5):
                    loss += lambda_weight*kl_to_identity_from_L(model.gaussians[i].make_L())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
                optimizer.step()
                print("loss at iteration ",k," is:",np.round(loss.item(),4))
                torch.save(model.state_dict(), weights_path)
    else:
        sampled_pixels = []
        for k in range(10):
            batch_image, _ = batched_import(batch_size=8)
            numpy_batch_image = torch.from_numpy(np.array(batch_image))
            filters = apply_filters(batch_image)
            batch_filters = torch.from_numpy(filters)
            indices_sample = []
            for i in range(8):
                indices_channel = []
                for j in range(3):
                    batch_filters_max = batch_filters[i,:,:,j].argmax()
                    idx = torch.unravel_index(batch_filters_max ,(512,512))
                    indices_channel.append(numpy_batch_image[i][idx[0],idx[1]])
                stacked_indices_channel = torch.stack(indices_channel)
                indices_sample.append(stacked_indices_channel) 
            indices_sample = torch.stack(indices_sample)
            sampled_pixels.append(indices_sample)
            print("batch processed:",k)
        sampled_pixels = (torch.stack(sampled_pixels).float()/255).mean(dim=0).mean(dim=0)
        for i in range(3):
            model.gaussians[i].mu.data=sampled_pixels[i]
        torch.save(model.state_dict(), weights_path)
            
        # batch_index = np.where((batch_filters.numpy()-batch_filters_max[:,None,None,:])**2<=1e-10)
        # print(batch_filters.numpy()[batch_index],batch_filters_max)
        # train_loop(model,optimizer,numpy_batch_image,device)
        # input('yipo')

if __name__ == "__main__":
    main()


