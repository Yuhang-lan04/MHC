import torch
from loss import Loss  
from TextNet import TextNet 
from MultiViewNet import MultiViewNet
from measure import *  
from test import test_MHC
import time  

def train_MHC(mul_X_train, train_label, original_train_labels, mul_X_rtest, yrt_label, device, args):
    image_models = [MultiViewNet(y_dim=X.shape[1], bit=args.n_z).to(device) for X in mul_X_train]
    text_model = TextNet(y_dim=len(train_label[0]), bit=args.n_z).to(device)
    loss_model = Loss().to(device)
    optimizers = [torch.optim.Adam(model.parameters(), lr=args.lr) for model in image_models]
    text_optimizer = torch.optim.Adam(text_model.parameters(), lr=args.lr)

    start_time = time.time()
    
    for epoch in range(int(args.maxiter)):
        text_model.train()
        for model in image_models:
            model.train()

        epoch_loss = 0.0
        num_batches = int(np.ceil(len(train_label) / args.batch_size))

        for batch_idx in range(num_batches):
            idx = np.arange(batch_idx * args.batch_size, min((batch_idx + 1) * args.batch_size, len(train_label)))
            label_batch = torch.tensor(train_label[idx], device=device, requires_grad=True)
            label_encodings_batch = text_model(label_batch)
            label_encodings_batch = label_encodings_batch.tanh()
            original_train_labels_batch = original_train_labels[idx]
            label_encodings_batch_mask = mask(label_encodings_batch, original_train_labels_batch, args.varphi)
            
            for optimizer in optimizers:
                optimizer.zero_grad()
            text_optimizer.zero_grad()

            Class_prompt_Contrastive_Learning = loss_model.class_prompt_contrastive_Learning(label_encodings_batch, original_train_labels_batch, args.temperature)

            mul_X_train_batch = [X[idx].to(device) for X in mul_X_train]

            fusion_z_list = [model(X_batch) for model, X_batch in zip(image_models, mul_X_train_batch)]
            fusion_z_list_encoding = [torch.tanh(z) for z in fusion_z_list]
            Supervised_Cross_view_Contrastive = loss_model.supervised_cross_view_contrastive(fusion_z_list_encoding, original_train_labels_batch, args.temperature)  # 拉近同一样本不同视图
            fusion_z_encoding = torch.mean(torch.stack(fusion_z_list), dim=0)

            Boundary_aware_Independent_Hashing = loss_model.boundary_aware_independent_hashing(fusion_z_encoding, label_encodings_batch_mask, original_train_labels_batch, args.xi)
            all_loss = Class_prompt_Contrastive_Learning+(Supervised_Cross_view_Contrastive * args.weight_supervised_cross_view_contrastive) + (Boundary_aware_Independent_Hashing * args.weight_boundary_aware_independent_hash)
            all_loss.backward()
            for optimizer in optimizers:
                optimizer.step()
            text_optimizer.step()

            epoch_loss += all_loss.item()
        
        if (epoch + 1) % 20 == 0 or epoch == args.maxiter - 1:
            with torch.no_grad():
                mul_X_encoded = torch.mean(torch.stack([model(X.to(device)) for model, X in zip(image_models, mul_X_train)]), dim=0)
                mul_X_hashes = torch.sign(mul_X_encoded)
                yy_pred, _ = test_MHC(image_models, mul_X_rtest, mul_X_hashes, original_train_labels, device, args)
                accuracy = do_metric(yy_pred, yrt_label)
                print(f"Epoch [{epoch+1}/{args.maxiter}] Average Loss: {epoch_loss / num_batches:.4f} ACC: {accuracy:.4f}")

                elapsed_time = time.time() - start_time
                print(f"Time taken for epochs {epoch-18} to {epoch+1}: {elapsed_time:.2f} seconds")
                start_time = time.time()

    return image_models, mul_X_hashes


def mask(label_encodings_batch, original_train_labels_batch, varphi):
    unique_labels = torch.unique(original_train_labels_batch)
    masked_encodings = torch.zeros_like(label_encodings_batch)
    for label in unique_labels:
        indices = [i for i, lbl in enumerate(original_train_labels_batch) if lbl == label]
        selected_encodings = label_encodings_batch[indices]
        sign_encodings = torch.sign(selected_encodings)
        avg_encoding = torch.abs(torch.mean(sign_encodings, axis=0))
        mask = (avg_encoding >= varphi).float()
        masked_encodings[indices] = selected_encodings * mask
    
    return masked_encodings