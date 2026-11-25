import json
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from rag.evaluation import evaluate, load_questions, load_sample_documents


class Command(BaseCommand):
    help = "Measure retrieval quality (recall@k, MRR) on the sample documents for several chunk sizes."

    def add_arguments(self, parser):
        parser.add_argument("--sizes", default="400,800,1200", help="comma-separated chunk sizes in characters")
        parser.add_argument("--overlap-ratio", type=float, default=0.15)
        parser.add_argument("--k", type=int, default=5)
        parser.add_argument("--out", default="results/eval.json")
        parser.add_argument("--min-recall", type=float, default=None, help="fail if the best recall@k is below this")

    def handle(self, *args, **opts):
        docs = load_sample_documents()
        questions = load_questions()
        runs = []
        for size in [int(s) for s in opts["sizes"].split(",")]:
            run = evaluate(docs, questions, size=size, overlap=int(size * opts["overlap_ratio"]), k=opts["k"])
            runs.append(run)
            self.stdout.write(
                f"chunk {run['chunk_size']:>5} / overlap {run['overlap']:>4}: {run['chunks']:>4} chunks, "
                f"recall@{run['k']} {run['hits']}/{run['questions']} ({run['recall_at_k']:.3f}), MRR {run['mrr']:.3f}"
            )
        out = Path(opts["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"documents": [d for d, _ in docs], "runs": runs}, indent=2) + "\n")
        best = max(r["recall_at_k"] for r in runs)
        if opts["min_recall"] is not None and best < opts["min_recall"]:
            raise CommandError(f"Best recall@{opts['k']} {best:.3f} is below the minimum {opts['min_recall']:.3f}")
