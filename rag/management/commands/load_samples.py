from django.core.management.base import BaseCommand

from rag.evaluation import SAMPLES_DIR
from rag.models import Document
from rag.services.ingest import ingest_bytes

TITLES = {
    "gdpr-ch01-02-principles.pdf": "GDPR Chapters I-II: General provisions and principles",
    "gdpr-ch03-data-subject-rights.pdf": "GDPR Chapter III: Rights of the data subject",
    "gdpr-ch04-controller-processor.docx": "GDPR Chapter IV: Controller and processor",
    "gdpr-ch05-international-transfers.pdf": "GDPR Chapter V: Transfers to third countries",
    "gdpr-ch08-remedies-penalties.md": "GDPR Chapter VIII: Remedies, liability and penalties",
}


class Command(BaseCommand):
    help = "Load the sample documents in samples/ (skips files that are already loaded)."

    def handle(self, *args, **opts):
        for path in sorted(SAMPLES_DIR.iterdir()):
            if path.suffix.lower() not in {".pdf", ".docx", ".txt", ".md"}:
                continue
            if Document.objects.filter(filename=path.name, is_sample=True).exists():
                continue
            doc = ingest_bytes(path.name, path.read_bytes(), title=TITLES.get(path.name), is_sample=True)
            self.stdout.write(f"loaded {path.name}: {doc.pages} pages, {doc.chunks.count()} chunks")
