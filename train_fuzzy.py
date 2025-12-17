import torch
import os
from model import GMM,FuzzyModel
from import_dataset import batched_import
from pre_process import apply_sobel_filter, binarize_objectness
import numpy as np
import gc


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
    weights_path = os.path.join(weights_dir, 'GMM.pt')
    os.makedirs(weights_dir, exist_ok=True)
    model_gmm = GMM(in_channels=3,num_gaussians=5).to(device)
    if os.path.exists(weights_path):
        model_gmm.load_state_dict(torch.load(weights_path))
        print(f"Loaded model weights from {weights_path}")
        model = FuzzyModel(model_gmm).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00001)
        lambda_weight = 1e-2
        for k in range(2):
            batch_image, batch_labels = batched_import(batch_size=2)
            processed_label_batch = binarize_objectness(batch_image,batch_labels)
            filter_image = apply_filters(batch_image)
            batch_image = (torch.from_numpy(np.concatenate([np.array(batch_image),filter_image],axis=3)).float()/255).to(device)
            pred = model.forward(batch_image)
            print(pred.shape)
            print(processed_label_batch.shape)
            input('yipi')
            # for i in range(8*512*512//2000):
            #     optimizer.zero_grad()
            #     batch = batch_image[i*2000:(i+1)*2000]
            #     pred = model.forward_loss(batch)
            #     loss = pred.mean()
            #     log_pi = torch.log_softmax(model.pi,dim=0)
            #     pi = torch.exp(log_pi)
            #     entropy_penalty = lambda_weight*torch.sum(pi*(log_pi-torch.log(torch.tensor(1/5).to(device))))
            #     loss += entropy_penalty
            #     for i in range(5):
            #         loss += lambda_weight*kl_to_identity_from_L(model.gaussians[i].make_L())
            #     loss.backward()
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            #     optimizer.step()
            #     print("loss at iteration ",k," is:",np.round(loss.item(),4))
            #     torch.save(model.state_dict(), weights_path)
    
         
if __name__ == "__main__":
    main()

