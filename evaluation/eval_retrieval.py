import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def evaluate_retrieval(model, dataset, batch_size=32, device='cuda'):
    model.eval()
    image_feats = []
    text_feats = []

    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            images = torch.stack([x[0] for x in batch]).to(device)
            texts = [x[1] for x in batch]  # Assume text already tokenized
            images_emb = model.encode_image(images)
            texts_emb = model.encode_text(texts)

            image_feats.append(images_emb.cpu().numpy())
            text_feats.append(texts_emb.cpu().numpy())

    image_feats = np.vstack(image_feats)
    text_feats = np.vstack(text_feats)

    similarity_matrix = cosine_similarity(image_feats, text_feats)

    recall_at_10 = compute_recall_at_k(similarity_matrix, k=10)
    mrr = compute_mrr(similarity_matrix)

    return {'Recall@10': recall_at_10, 'MRR': mrr}


def compute_recall_at_k(sim_matrix, k=10):
    correct = 0
    for i in range(len(sim_matrix)):
        top_k = np.argsort(sim_matrix[i])[::-1][:k]
        if i in top_k:
            correct += 1
    return correct / len(sim_matrix)


def compute_mrr(sim_matrix):
    ranks = []
    for i in range(len(sim_matrix)):
        ranking = np.argsort(sim_matrix[i])[::-1]
        rank = np.where(ranking == i)[0][0] + 1
        ranks.append(1.0 / rank)
    return np.mean(ranks)
