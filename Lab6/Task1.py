import argparse
from collections import Counter, defaultdict
from pprint import pprint

def load_play_tennis():
    """Завантаження прикладу датасету 'Play Tennis'."""
    return [
        {"outlook": "Sunny", "humidity": "High", "wind": "Weak", "play": "No"},
        {"outlook": "Sunny", "humidity": "High", "wind": "Strong", "play": "No"},
        {"outlook": "Overcast", "humidity": "High", "wind": "Weak", "play": "Yes"},
        {"outlook": "Rain", "humidity": "High", "wind": "Weak", "play": "Yes"},
        {"outlook": "Rain", "humidity": "Normal", "wind": "Weak", "play": "Yes"},
        {"outlook": "Rain", "humidity": "Normal", "wind": "Strong", "play": "No"},
        {"outlook": "Overcast", "humidity": "Normal", "wind": "Strong", "play": "Yes"},
        {"outlook": "Sunny", "humidity": "High", "wind": "Weak", "play": "No"},
        {"outlook": "Sunny", "humidity": "Normal", "wind": "Weak", "play": "Yes"},
        {"outlook": "Rain", "humidity": "High", "wind": "Weak", "play": "Yes"},
        {"outlook": "Sunny", "humidity": "Normal", "wind": "Strong", "play": "Yes"},
        {"outlook": "Overcast", "humidity": "High", "wind": "Strong", "play": "Yes"},
        {"outlook": "Overcast", "humidity": "Normal", "wind": "Weak", "play": "Yes"},
        {"outlook": "Rain", "humidity": "High", "wind": "Strong", "play": "No"},
    ]

def build_freq_tables(rows, feature_names, target_name="play"):
    """Побудова частотних таблиць для ознак та класів."""
    class_counts = Counter(r[target_name] for r in rows)
    feat_counts = {f: {c: Counter() for c in class_counts} for f in feature_names}
    value_domains = {f: set() for f in feature_names}

    for r in rows:
        c = r[target_name]
        for f in feature_names:
            v = r[f]
            feat_counts[f][c][v] += 1
            value_domains[f].add(v)

    return class_counts, feat_counts, {f: sorted(list(vs)) for f, vs in value_domains.items()}

def likelihood_tables(class_counts, feat_counts, value_domains, alpha=0.0):
    """Обчислення таблиць ймовірностей з Лапласовим згладжуванням."""
    like = defaultdict(lambda: defaultdict(dict))
    for f, per_class in feat_counts.items():
        K = len(value_domains[f])
        for c, counts in per_class.items():
            total_c = class_counts[c]
            for v in value_domains[f]:
                num = counts.get(v, 0) + alpha
                den = total_c + alpha * K
                like[f][c][v] = num / den
    return like

def posterior(x, class_counts, like_table):
    """Обчислення апостеріорних ймовірностей P(C|x)."""
    N = sum(class_counts.values())
    scores = {}
    for c, cnt in class_counts.items():
        prior = cnt / N
        cond = 1.0
        for f, v in x.items():
            cond *= like_table[f][c].get(v, 0.0)
        scores[c] = prior * cond

    s = sum(scores.values())
    if s > 0:
        for c in scores:
            scores[c] /= s
    return scores

def main():
    parser = argparse.ArgumentParser(description="Наївний Байєс для датасету 'Play Tennis'")
    parser.add_argument("--alpha", type=float, default=0.0,
                        help="Лапласове згладжування (0 = без)")
    args = parser.parse_args()

    rows = load_play_tennis()
    feature_names = ["outlook", "humidity", "wind"]

    class_counts, feat_counts, domains = build_freq_tables(rows, feature_names)
    like = likelihood_tables(class_counts, feat_counts, domains, alpha=args.alpha)

    print("Кількість класів (prior counts):", class_counts)
    print("\nДомен ознак:")
    pprint(domains)

    print("\nLikelihood-таблиці (P(feature=value | class)):")
    pprint({f: {c: dict(d) for c, d in like[f].items()} for f in feature_names})

    x = {"outlook": "Rain", "humidity": "High", "wind": "Weak"}
    post = posterior(x, class_counts, like)
    print("\nПриклад (Rain, High, Weak): апостеріорні P(C|x):", post)
    pred = max(post, key=post.get)
    print("Прогноз:", pred)


if __name__ == "__main__":
    main()