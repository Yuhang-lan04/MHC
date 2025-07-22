import torch
import time


def test_MHC(models, mul_X_rtest, mul_X_hashes, labels, device, args):
    for model in models:
        model.eval()

    num_X_test = mul_X_rtest[0].shape[0]
    bit = models[0].fc[-1].out_features
    tmp_q = torch.zeros([num_X_test, bit]).to(device)
    
    start_time = time.time()
    batch_size = args.batch_size
    num_batches = (num_X_test + batch_size - 1) // batch_size  # 计算批次数量，向上取整
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_X_test)
        idx = torch.arange(start_idx, end_idx)

        mul_X_test_batch = [X[idx].to(device) for X in mul_X_rtest]

        latent_representations = [model(x).detach() for model, x in zip(models, mul_X_test_batch)]

        latent_representations_z = torch.mean(torch.stack(latent_representations), dim=0)
        hash_codes = torch.sign(latent_representations_z)
        tmp_q[idx] = hash_codes

    distances = 0.5 * (bit - torch.matmul(tmp_q, mul_X_hashes.t()))
    min_distance_indices = torch.argmin(distances, dim=1)

    yy_pred = labels[min_distance_indices]
    end_time = time.time()

    return yy_pred, end_time - start_time



