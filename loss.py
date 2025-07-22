import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def class_prompt_contrastive_Learning(self, label_encodings_batch, original_train_labels_batch, temperature):
        cos_sim_matrix = torch.nn.functional.cosine_similarity(
            label_encodings_batch.unsqueeze(1),
            label_encodings_batch.unsqueeze(0),
            dim=-1
        )

        exp_sim_matrix = torch.exp(cos_sim_matrix / temperature).clamp(max=1e10)
        label_equal_mask = original_train_labels_batch.unsqueeze(1) == original_train_labels_batch.unsqueeze(0)

        numerator = exp_sim_matrix * label_equal_mask
        denominator = exp_sim_matrix
        denominator_sum = denominator.sum(dim=1).clamp(min=1e-10)

        loss = -torch.log(numerator.sum(dim=1) / denominator_sum)
        return loss.mean()

    def supervised_cross_view_contrastive(self, fusion_z_list, original_train_labels_batch, temperature):
        M = len(fusion_z_list)

        fusion_z = torch.cat(fusion_z_list, dim=0)
        original_train_labels_expanded = original_train_labels_batch.repeat(M)

        label_equal_mask = original_train_labels_expanded.unsqueeze(1) == original_train_labels_expanded.unsqueeze(0)

        sim_matrix = torch.nn.functional.cosine_similarity(fusion_z.unsqueeze(1), fusion_z.unsqueeze(0),dim=2)

        numerator = torch.exp(sim_matrix / temperature) * label_equal_mask
        numerator_sum = numerator.sum(dim=1)
        denominator = torch.exp(sim_matrix / temperature).sum(dim=1)
        loss = -torch.log(numerator_sum / denominator.clamp(min=1e-10))
        return loss.mean()


    def boundary_aware_independent_hashing(self, sample_embeddings, label_embeddings, labels, xi):
        cos_sim_matrix = torch.nn.functional.cosine_similarity(sample_embeddings.unsqueeze(1),
                                                               label_embeddings.unsqueeze(0), dim=2)
        D_matrix = (1 - cos_sim_matrix) / 2

        labels_equal = labels.unsqueeze(1) == labels.unsqueeze(0)
        y_matrix = labels_equal.float().to(sample_embeddings.device)

        loss_matrix = y_matrix * D_matrix ** 2 + (1 - y_matrix) * torch.clamp(xi - D_matrix, min=0) ** 2

        return loss_matrix.mean()


