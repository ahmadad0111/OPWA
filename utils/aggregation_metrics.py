import torch
import torch.nn.functional as F

def compute_distances(embeddings):
    # Compute pairwise Euclidean distances between embeddings
    diff = embeddings.unsqueeze(1) - embeddings.unsqueeze(0)
    distances = torch.sqrt(torch.sum(diff ** 2, dim=2))
    return distances

def compute_iter_class_distances(class_embeddings, mean_embeddings,class_idx):
    # Compute pairwise Euclidean distances between embeddings
    total_distances = torch.zeros((len(mean_embeddings),len(mean_embeddings)),device=class_embeddings.device)

    for c in range(4):
        if c != class_idx:
            diff = class_embeddings.unsqueeze(1) - mean_embeddings[:, c, :].unsqueeze(0)
            distances = torch.sqrt(torch.sum(diff ** 2, dim=2))
            #distances = 1 / (distances + 1e-10)
            total_distances += distances
        # else:
        #     diff = class_embeddings.unsqueeze(1) - mean_embeddings[:, c, :].unsqueeze(0)
        #     distances = torch.sqrt(torch.sum(diff ** 2, dim=2))
        #     total_distances += distances
    return total_distances



def compute_weights(distances, epsilon=1e-10):
    # Inverse distance weighting
    #weights = 1 / (distances + epsilon)

    # use gaussian kernel distance
    sigma = 0.5
    weights = torch.exp(-distances**2 / (2 * sigma**2))
    return weights
def compute_inter_class_weights(distances, epsilon=1e-10):
    # Inverse distance weighting
    #distances = 1 / (distances + epsilon)

    # use gaussian kernel distance
    sigma = 1
    weights = torch.exp(-distances**2 / (2 * sigma**2))
    return weights




def aggregate_embeddings(mean_embeddings, weights):
    # # Aggregate embeddings using weights

    
    weight_sum = weights.sum(dim=1, keepdim=True)
    W_norm = weights / weight_sum

    consensus_embeddings = torch.mm(W_norm, mean_embeddings)
    
    
    return consensus_embeddings.mean(dim=0)

def normalize(matrix):
    return (matrix - matrix.min()) / (matrix.max() - matrix.min())


def compute_softmax_weights(distances):
    return F.softmax(distances, dim=-1)

def aggregate_subj_embeddings(mean_embeddings):
    # # Example usage:
    num_subjects = mean_embeddings.size(0)
    num_classes = mean_embeddings.size(1)
    embedding_dim = mean_embeddings.size(2)

    # # Generate random embeddings for illustration
    # mean_embeddings = torch.rand(num_subjects, num_classes, embedding_dim)

    # Calculate pairwise distances and weights
    distances = torch.zeros((num_classes, num_subjects, num_subjects),device=mean_embeddings.device)
    weights = torch.zeros((num_classes, num_subjects,num_subjects),device=mean_embeddings.device)

    for class_idx in range(num_classes):
        class_embeddings = mean_embeddings[:, class_idx, :]
        # inter class distance
        class_distances = compute_distances(class_embeddings)
        inter_class_distances = compute_iter_class_distances(class_embeddings,mean_embeddings,class_idx)
        #inter_class_distances = normalize(inter_class_distances)
        #inter_class_weights = compute_inter_class_weights(inter_class_distances)
        distances[class_idx] = inter_class_distances
        #weights[class_idx] = inter_class_weights
        #print(weights[class_idx])

        weights[class_idx] = compute_softmax_weights(inter_class_distances)


        # # # class_distances = compute_distances(class_embeddings)
        class_weights = compute_weights(class_distances)
        #print(class_weights)
        distances[class_idx] = class_distances
        weights[class_idx] += class_weights


    # Aggregate embeddings for each class
    consensus_embeddings = torch.zeros((num_classes, embedding_dim),device=mean_embeddings.device)

    for class_idx in range(num_classes):
        class_weights = weights[class_idx]/2
        class_mean_embeddings = mean_embeddings[:, class_idx, :]
        consensus_embeddings[class_idx] = aggregate_embeddings(class_mean_embeddings, class_weights)
    return consensus_embeddings


# def compute_weights(distances, epsilon=1e-10):
#     # Inverse distance weighting
#     weights = 1 / (distances + epsilon)
#     # Set diagonal to zero to ignore self-distances
#     weights.fill_diagonal_(0)
#     return torch.sum(weights, dim=1)
# def aggregate_embeddings(mean_embeddings, weights):
#     # Aggregate embeddings using weights
#     weight_sums = torch.sum(weights)
#     weights = weights.unsqueeze(1).repeat(1,128)

#     weighted_sum = mean_embeddings * weights
    
#     weighted_sum = torch.sum(weighted_sum, dim=0)

#     # Avoid division by zero
#     consensus_embeddings = weighted_sum / (weight_sums)


#     return consensus_embeddings


# def aggregate_subj_embeddings(mean_embeddings):
#     # # Example usage:
#     num_subjects = len(mean_embeddings)
#     num_classes = 4#len(mean_embeddings[1])
#     embedding_dim = 128#len(mean_embeddings[2])

#     # # Generate random embeddings for illustration
#     # mean_embeddings = torch.rand(num_subjects, num_classes, embedding_dim)

#     # Calculate pairwise distances and weights
#     distances = torch.zeros((num_classes, num_subjects, num_subjects),device=mean_embeddings.device)
#     weights = torch.zeros((num_classes, num_subjects),device=mean_embeddings.device)

#     for class_idx in range(num_classes):
#         class_embeddings = mean_embeddings[:, class_idx, :]
#         class_distances = compute_distances(class_embeddings)
#         class_weights = compute_weights(class_distances)
        
#         distances[class_idx] = class_distances
#         weights[class_idx] = class_weights

#     # Aggregate embeddings for each class
#     consensus_embeddings = torch.zeros((num_classes, embedding_dim),device=mean_embeddings.device)

#     for class_idx in range(num_classes):
#         class_weights = weights[class_idx]
#         class_mean_embeddings = mean_embeddings[:, class_idx, :]
#         consensus_embeddings[class_idx] = aggregate_embeddings(class_mean_embeddings, class_weights)
#     return consensus_embeddings