import torch

@torch.no_grad()
def get_max_confidence_and_residual_variance(predictions, valid_mask, num_classes, epsilon=1e-8):
    # predictions: [n, c, w, h]
    # valid_mask: [n, w, h]
    # num_classes: K (total number of classes)
    
    # Step 1: Expand valid_mask to match predictions' shape
    valid_mask_expanded = valid_mask.unsqueeze(1).expand_as(predictions)  # [n, c, w, h]

    # Step 2: Zero-fill invalid locations (no NaN) and compute max on valid entries only
    neg_inf = torch.finfo(predictions.dtype).min
    predictions_masked = torch.where(valid_mask_expanded == 1, predictions, torch.full_like(predictions, neg_inf))

    # Step 3: Calculate the maximum confidence and corresponding class (only over valid entries)
    max_confidence, max_indices = torch.max(predictions_masked, dim=1)  # [n, w, h]
    
    # Step 4: Compute the coefficient g_j
    g_j = (num_classes - 1)**2 / (2 * (1 - max_confidence + epsilon))  # [n, w, h]

    # Step 5: Create a mask to exclude the maximum confidence class
    one_hot_max = torch.nn.functional.one_hot(max_indices, num_classes=predictions.shape[1])  # [n, w, h, c]
    one_hot_max = one_hot_max.permute(0, 3, 1, 2)  # [n, c, w, h]
    
    # Step 6: Exclude the maximum prediction via mask (no NaNs)
    remaining_mask = valid_mask_expanded * (1 - one_hot_max)  # [n, c, w, h]
    remaining_predictions = predictions_masked * remaining_mask
    
    # Step 7: Compute mean over remaining classes (masked average)
    count_remaining = remaining_mask.sum(dim=1).clamp_min(1)  # [n, w, h]
    sum_remaining = remaining_predictions.sum(dim=1)  # [n, w, h]
    mean_remaining_predictions = sum_remaining / count_remaining  # [n, w, h]
    
    # Step 8: Calculate variance over remaining classes (mean of squared deviation, masked)
    diff = remaining_predictions - mean_remaining_predictions.unsqueeze(1)  # [n, c, w, h]
    sq_diff = diff ** 2 * remaining_mask
    sum_sq_diff = sq_diff.sum(dim=1)  # [n, w, h]
    residual_variance = sum_sq_diff / count_remaining  # [n, w, h]

    # Step 9: Scale residual variance by g_j
    scaled_residual_variance = g_j * residual_variance  # [n, w, h]
    # Zero out invalid pixels for stability
    max_confidence = torch.where(valid_mask == 1, max_confidence, torch.zeros_like(max_confidence))
    scaled_residual_variance = torch.where(valid_mask == 1, scaled_residual_variance, torch.zeros_like(scaled_residual_variance))
    
    return max_confidence, scaled_residual_variance

@torch.no_grad()
def batch_class_stats(max_conf, res_var):
    means = []
    vars = []
    for index in range(max_conf.shape[0]):
        features = torch.stack([max_conf[index], res_var[index]], dim=-1).view(-1, 2)  # [w*h, 2]
        valid_mask = ~torch.isnan(features).any(dim=-1)
        valid_features = features[valid_mask]

        if valid_features.size(0) == 0:
            means.append(torch.tensor((1, 0), device=max_conf.device))
            vars.append(torch.tensor((1, 1), device=max_conf.device)) 
            continue
        class_assignments = _class_assignment(valid_features, 2)
        class_centers = _compute_class_centers(valid_features, class_assignments, 2)
        max_mean_idx = torch.argmax(class_centers[0][:, 0])  
        selected_mean = class_centers[0][max_mean_idx] 
        selected_var = class_centers[1][max_mean_idx] 
        means.append(selected_mean)
        vars.append(selected_var)
    return torch.stack(means), torch.stack(vars)

@torch.no_grad()
def _compute_eigenvectors_with_svd(X, num_classes):
    U, S, Vt = torch.linalg.svd(X.T, full_matrices=False)
    eigvals = S ** 2 
    idx = torch.argsort(-eigvals) 
    eigvecs = Vt.T[:, idx[:num_classes]]  
    return eigvecs

@torch.no_grad()
def _class_assignment(input, num_classes):
    eigenvectors = _compute_eigenvectors_with_svd(input, num_classes)
    class_assignments = torch.argmax(torch.abs(eigenvectors), dim=1)
    return class_assignments

@torch.no_grad()
def _compute_class_centers(features, class_assignments, num_classes):
    means = []
    vars = []
    for class_id in range(num_classes):
        points_in_class = features[class_assignments == class_id]
        num_points = points_in_class.size(0)
        if num_points == 0:
            mean = torch.zeros(features.size(1), device=features.device)
            var = torch.zeros(features.size(1), device=features.device)
        elif num_points == 1:
            mean = points_in_class.squeeze(0)
            var = torch.zeros(features.size(1), device=features.device)
        else:
            mean = points_in_class.mean(dim=0)
            var = points_in_class.var(dim=0, unbiased=True)
        means.append(mean)
        vars.append(var)
    return torch.stack(means), torch.stack(vars)
