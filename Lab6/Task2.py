import argparse
from collections import Counter, defaultdict

ROWS = [
    {"outlook":"Sunny","humidity":"High","wind":"Weak","play":"No"},
    {"outlook":"Sunny","humidity":"High","wind":"Strong","play":"No"},
    {"outlook":"Overcast","humidity":"High","wind":"Weak","play":"Yes"},
    {"outlook":"Rain","humidity":"High","wind":"Weak","play":"Yes"},
    {"outlook":"Rain","humidity":"Normal","wind":"Weak","play":"Yes"},
    {"outlook":"Rain","humidity":"Normal","wind":"Strong","play":"No"},
    {"outlook":"Overcast","humidity":"Normal","wind":"Strong","play":"Yes"},
    {"outlook":"Sunny","humidity":"High","wind":"Weak","play":"No"},
    {"outlook":"Sunny","humidity":"Normal","wind":"Weak","play":"Yes"},
    {"outlook":"Rain","humidity":"High","wind":"Weak","play":"Yes"},
    {"outlook":"Sunny","humidity":"Normal","wind":"Strong","play":"Yes"},
    {"outlook":"Overcast","humidity":"High","wind":"Strong","play":"Yes"},
    {"outlook":"Overcast","humidity":"Normal","wind":"Weak","play":"Yes"},
    {"outlook":"Rain","humidity":"High","wind":"Strong","play":"No"},
]

FEATURES = ["outlook", "humidity", "wind"]

COND = {"outlook": "Rain", "humidity": "High", "wind": "Strong"}

def build_counts(rows):
    class_counts = Counter(r["play"] for r in rows)
    feat_counts = {f: {c: Counter() for c in class_counts} for f in FEATURES}
    domains = {f: set() for f in FEATURES}
    for r in rows:
        c = r["play"]
        for f in FEATURES:
            v = r[f]
            feat_counts[f][c][v] += 1
            domains[f].add(v)
    domains = {f: sorted(list(vs)) for f, vs in domains.items()}
    return class_counts, feat_counts, domains

def likelihood_tables(class_counts, feat_counts, domains, alpha=1.0):
    like = defaultdict(lambda: defaultdict(dict))
    for f, per_class in feat_counts.items():
        K = len(domains[f])
        for c, counts in per_class.items():
            total_c = class_counts[c]
            for v in domains[f]:
                num = counts.get(v, 0) + alpha
                den = total_c + alpha * K
                like[f][c][v] = num / den
    return like

def posterior(x, class_counts, like):
    N = sum(class_counts.values())
    scores = {}
    for c, cnt in class_counts.items():
        prior = cnt / N
        cond = 1.0
        for f, v in x.items():
            cond *= like[f][c].get(v, 0.0)
        scores[c] = prior * cond
    s = sum(scores.values())
    if s > 0:
        for c in scores:
            scores[c] /= s
    return scores

def main():
    ap = argparse.ArgumentParser(description="NB для варіантів 5,10,15 (Rain, High, Strong)")
    ap.add_argument("--alpha", type=float, default=1.0, help="Лапласове згладжування (0=без, 1=add-one)")
    ap.add_argument("--no-header", action="store_true", help="не друкувати заголовок варіантів")
    args = ap.parse_args()

    class_counts, feat_counts, domains = build_counts(ROWS)
    like = likelihood_tables(class_counts, feat_counts, domains, alpha=args.alpha)

    post = posterior(COND, class_counts, like)
    pred = max(post, key=post.get)

    if not args.no_header:
        print("Варіанти: 5, 10, 15 — умова Outlook=Rain, Humidity=High, Wind=Strong")
    print(f"alpha={args.alpha}")
    print(f"P(Yes|x) = {post.get('Yes',0):.6f}")
    print(f"P(No |x) = {post.get('No',0):.6f}")
    print(f"Прогноз : {pred}")

if __name__ == "__main__":
    main()