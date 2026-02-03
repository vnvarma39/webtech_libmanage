from rouge_score import rouge_scorer

def evaluate_summary(reference, generated):
    sc = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scores = sc.score(reference, generated)
    return {k: round(v.fmeasure, 3) for k, v in scores.items()}