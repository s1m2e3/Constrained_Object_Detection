import torch
import os
from model import GMM,FuzzyModel
from import_dataset import batched_import
from pre_process import apply_sobel_filter, binarize_objectness
import numpy as np
import gc
import matplotlib.pyplot as plt

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
        np.stack(flatten(sobel_processed := apply_sobel_filter(image_batch)) ,axis=0),
    ),axis=3)
    return all_processed



def main():
    gc.collect()
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights_dir = 'weights'
    weights_gmm_path = os.path.join(weights_dir, 'GMM.pt')
    os.makedirs(weights_dir, exist_ok=True)
    model_gmm = GMM(in_channels=3,num_gaussians=5).to(device)
    weights_fuzzy_path = os.path.join(weights_dir, 'fuzzy_model.pt')
    if os.path.exists(weights_gmm_path):
        model_gmm.load_state_dict(torch.load(weights_gmm_path))
        print(f"Loaded model weights from {weights_gmm_path}")
    model = FuzzyModel(model_gmm).to(device)
    if os.path.exists(weights_fuzzy_path):
        model.load_state_dict(torch.load(weights_fuzzy_path))
        print(f"Loaded model weights from {weights_fuzzy_path}")
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00001)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9,weight_decay=0.0001)

    lambda_weight = 1e-4
    loss = torch.nn.BCEWithLogitsLoss(weight=(torch.ones(512)*10).to(device))
    for k in range(1000):
        batch_image, batch_labels = batched_import(batch_size=2)
        processed_label_batch = torch.from_numpy(binarize_objectness(batch_image,batch_labels)).to(device)
        pos_index = torch.where(processed_label_batch==1)
        mask_pos = (processed_label_batch==1)
        if len(pos_index[0])>100:
            filter_image = apply_filters(batch_image)
            tensor_image = (torch.from_numpy(np.concatenate([np.array(batch_image),filter_image],axis=3)).float()/255).to(device)
            image_tensor_pixels = torch.from_numpy((np.array(batch_image)).reshape(2*512*512,3)/255).to(device).float()
            pred = model.forward(tensor_image)
            mask_neg = (processed_label_batch==0)
            pred_constrained = model.eval_constraints(pred,mask_pos,mask_neg)

            # pred_centroids_distances = model.centroid_distances(tensor_image)
            # pr/ed_centroids = model.centroid(tensor_image)
            # pos_neg = torch.where(processed_label_batch==0)
            
            pred_normalized = model.normalize(pred_constrained)
            
            # print(pred_constrained[pos_index].max().item(), pred_constrained[pos_index].min().item(), pred_constrained[pos_index].mean().item())
            # print(pred_constrained[pos_neg].max().item(), pred_constrained[pos_neg].min().item(), pred_constrained[pos_neg].mean().item())
            # input('yipo')
            # pred_labels = torch.sigmoid(pred)
            processed_label_batch_all = processed_label_batch.amax(dim=1)
            pred_labels_all = pred_normalized.amax(dim=1)
            
            fig, ax = plt.subplots(
                3, 2,
                figsize=(10, 24),                 # taller figure so each row is larger
                constrained_layout=True,         # smarter layout engine
                sharex=True, sharey=True
            )
            # kill inter-axes gaps
            fig.set_constrained_layout_pads(w_pad=0.0, h_pad=0.0, wspace=0.0, hspace=0.0)

            for i in range(2):
                ax[0, i].imshow(processed_label_batch_all[i].cpu().detach().numpy(), cmap='gray')
                ax[1, i].imshow(pred_labels_all[i].cpu().detach().numpy(), cmap='gray')
                ax[2, i].imshow(np.array(batch_image)[i])

            # # remove all ticks and frames; also avoid extra margins
            for a in ax.ravel():
                a.set_axis_off()
                a.set_aspect('equal')   # keep correct geometry; use 'auto' to stretch if desired
                a.margins(0)

            plt.show()
            loss_bce = loss(pred,processed_label_batch) + ((pred - pred_constrained)**2).mean()
            
            pred_pixel = model.gmm.forward_loss(image_tensor_pixels).mean()
            print("loss at iteration ",k," is:",np.round(loss_bce.item(),6))
            loss_centroids = -model.gmm.centroid_loss()
            loss_bce = loss_bce+pred_pixel+lambda_weight*loss_centroids
            loss_bce.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            with torch.no_grad():
                for gaussian in model.gmm.gaussians:
                    gaussian.mu.clamp_(0.0,1.0)
                    print(gaussian.mu.data)
                    
            torch.save(model.state_dict(), weights_fuzzy_path)
    
if __name__ == "__main__":
    main()
    
