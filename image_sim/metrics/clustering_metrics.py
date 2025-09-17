def exact_match(true_groups: list, pred_clusters: list) -> float:
    true_sets = [set(group) for group in true_groups]
    pred_sets = [set(cluster) for cluster in pred_clusters]

    print(f"Совпавшие группы: ")
    correct = 0
    for p in pred_sets:
        if p in true_sets:
            correct += 1
            print(p)
    return correct / len(true_sets)


def partial_match(true_groups: list, pred_clusters: list, threshold: float = 0.8):
    true_sets = [set(group) for group in true_groups]
    correct = 0

    for p in pred_clusters:
        p_set = set(p)
        max_overlap = max(len(p_set & t) / len(p_set) for t in true_sets)
        if max_overlap >= threshold:
            correct += 1

    return correct / len(true_sets)