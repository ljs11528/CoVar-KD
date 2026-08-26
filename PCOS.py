import torch

from utils.covar_metrics import covar_components_from_probabilities

@torch.no_grad()
def get_max_confidence_and_residual_variance_components(predictions, valid_mask, num_classes=None, epsilon=1e-8,
                                                       a=None):
    if num_classes is None:
        num_classes = predictions.shape[1]
    if int(num_classes) != int(predictions.shape[1]):
        raise ValueError("num_classes must match the probability class dimension")

    components = covar_components_from_probabilities(
        predictions,
        class_dim=1,
        coefficient_a=a,
        epsilon=epsilon,
    )
    valid = valid_mask.bool()
    max_confidence = torch.where(valid, components["confidence"], torch.zeros_like(components["confidence"]))
    residual_variance = torch.where(
        valid,
        components["residual_variance"],
        torch.zeros_like(components["residual_variance"]),
    )
    scaled_residual_variance = torch.where(
        valid,
        components["r_v"],
        torch.zeros_like(components["r_v"]),
    )
    return max_confidence, residual_variance, scaled_residual_variance

@torch.no_grad()
def get_max_confidence_and_residual_variance(predictions, valid_mask, num_classes=None, epsilon=1e-8, a=None):
    max_confidence, _, scaled_residual_variance = get_max_confidence_and_residual_variance_components(
        predictions, valid_mask, num_classes=num_classes, epsilon=epsilon, a=a
    )
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
