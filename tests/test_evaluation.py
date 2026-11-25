import pytest
from django.core.management import call_command

from rag.evaluation import evaluate, label_chunks, load_questions, load_sample_documents
from rag.models import Document
from rag.services.chunking import TextChunk


def chunk(i, text):
    return TextChunk(index=i, page=1, start_char=0, text=text)


def test_labels_follow_the_open_article():
    chunks = [
        chunk(0, "Article 5 - Principles\n1. Personal data shall be processed lawfully."),
        chunk(1, "kept no longer than necessary.\nArticle 6 - Lawfulness of processing\n1. Processing is lawful"),
        chunk(2, "only if one of the following applies"),
        chunk(3, "Article 7 - Conditions for consent\nWhere processing is based on consent"),
    ]
    assert label_chunks(chunks) == [{5}, {5, 6}, {6}, {7}]


def test_retrieval_quality_on_the_sample_documents():
    docs = load_sample_documents()
    questions = load_questions()
    assert len(docs) == 5 and len(questions) == 25
    result = evaluate(docs, questions, size=1200, overlap=180, k=5)
    assert result["recall_at_k"] >= 0.88  # 22 of 25 or better
    assert result["mrr"] >= 0.6


@pytest.mark.django_db
def test_load_samples_is_idempotent():
    call_command("load_samples")
    call_command("load_samples")
    assert Document.objects.filter(is_sample=True).count() == 5
    assert {d.file_type for d in Document.objects.all()} == {"pdf", "docx", "md"}
