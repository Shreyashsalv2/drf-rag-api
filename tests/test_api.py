import json

import pytest
import responses
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework.authtoken.models import Token
from rest_framework.throttling import ScopedRateThrottle

from rag.models import Chunk, Document
from rag.services import vectorstore
from tests.conftest import BREACH_TEXT, CATS_TEXT, make_docx, make_pdf

pytestmark = pytest.mark.django_db

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"


def upload(api, name, data, **extra):
    return api.post("/api/documents/", {"file": SimpleUploadedFile(name, data), **extra}, format="multipart")


@pytest.fixture
def two_docs(api):
    breach = upload(api, "breach-rules.txt", BREACH_TEXT.encode()).json()
    cats = upload(api, "cat-care.md", CATS_TEXT.encode()).json()
    return breach, cats


class TestDocuments:
    def test_upload_text_file(self, api):
        res = upload(api, "breach_rules.txt", BREACH_TEXT.encode(), title="Breach rules")
        assert res.status_code == 201
        body = res.json()
        assert body["title"] == "Breach rules" and body["file_type"] == "txt"
        assert body["pages"] == 1 and body["chunk_count"] >= 1 and body["size_bytes"] == len(BREACH_TEXT)
        assert vectorstore.count() == body["chunk_count"]

    def test_upload_pdf_and_docx(self, api):
        data = make_pdf(["Page one text about consent.", "Page two text about erasure."])
        pdf = upload(api, "two-pages.pdf", data).json()
        assert pdf["file_type"] == "pdf" and pdf["pages"] == 2
        chunks = api.get(f"/api/documents/{pdf['id']}/chunks/").json()
        assert [c["page"] for c in chunks["results"]] == [1, 2]
        doc = upload(api, "memo.docx", make_docx(["A memo about retention periods."])).json()
        assert doc["file_type"] == "docx" and doc["title"] == "memo"

    def test_list_and_retrieve(self, api, two_docs):
        listing = api.get("/api/documents/").json()
        assert listing["count"] == 2
        assert {d["filename"] for d in listing["results"]} == {"breach-rules.txt", "cat-care.md"}
        one = api.get(f"/api/documents/{two_docs[0]['id']}/").json()
        assert one["title"] == "breach rules"
        assert api.get("/api/documents/9999/").status_code == 404

    @pytest.mark.parametrize(
        "name,data,code",
        [
            ("photo.png", b"\x89PNG", "invalid_request"),
            ("empty.txt", b"", "invalid_request"),
            ("broken.pdf", b"%PDF-1.4 garbage", "unreadable_file"),
        ],
    )
    def test_bad_uploads_get_a_json_error(self, api, name, data, code):
        res = upload(api, name, data)
        assert res.status_code == 400
        assert res.json()["error"]["code"] == code
        assert Document.objects.count() == 0

    def test_too_large_upload(self, api, settings):
        settings.RAG = {**settings.RAG, "MAX_UPLOAD_MB": 0.0001}
        res = upload(api, "big.txt", b"x" * 500)
        assert res.status_code == 400
        assert "larger than" in json.dumps(res.json())

    def test_uploads_are_throttled(self, api, monkeypatch):
        monkeypatch.setattr(ScopedRateThrottle, "THROTTLE_RATES", {"uploads": "2/min"})
        codes = [upload(api, f"n{i}.txt", BREACH_TEXT.encode()).status_code for i in range(3)]
        assert codes == [201, 201, 429]

    def test_delete_needs_a_token_and_removes_vectors(self, api, two_docs):
        breach_id = two_docs[0]["id"]
        assert api.delete(f"/api/documents/{breach_id}/").status_code == 401
        token = Token.objects.create(user=User.objects.create_user("admin", password="unused-in-tests"))
        api.credentials(HTTP_AUTHORIZATION=f"Token {token.key}")
        assert api.delete(f"/api/documents/{breach_id}/").status_code == 204
        assert not Chunk.objects.filter(document_id=breach_id).exists()
        results = api.post("/api/search/", {"query": "breach notification 72 hours"}, format="json").json()["results"]
        assert all(r["document_id"] != breach_id for r in results)
        assert vectorstore.count() == Chunk.objects.count()


class TestSearchAndAsk:
    def test_search_ranks_the_relevant_document_first(self, api, two_docs):
        res = api.post("/api/search/", {"query": "How fast must a data breach be reported?", "k": 5}, format="json")
        assert res.status_code == 200
        results = res.json()["results"]
        assert results[0]["document_id"] == two_docs[0]["id"]
        assert results[0]["page"] == 1 and "72 hours" in results[0]["text"]
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_can_be_limited_to_documents(self, api, two_docs):
        cats_id = two_docs[1]["id"]
        res = api.post("/api/search/", {"query": "data breach", "document_ids": [cats_id]}, format="json")
        assert {r["document_id"] for r in res.json()["results"]} == {cats_id}

    @pytest.mark.parametrize("payload", [{"query": "a"}, {"query": "breach", "k": 50}, {"k": 3}, {"query": "x" * 501}])
    def test_search_validation(self, api, payload):
        res = api.post("/api/search/", payload, format="json")
        assert res.status_code == 400
        assert res.json()["error"]["code"] == "invalid_request"

    def test_ask_without_a_key_gives_an_extractive_cited_answer(self, api, two_docs):
        question = {"query": "When must the supervisory authority be notified of a breach?"}
        body = api.post("/api/ask/", question, format="json").json()
        assert body["mode"] == "extractive"
        assert "72 hours" in body["answer"] and "[1]" in body["answer"]
        assert body["sources"][0]["n"] == 1 and body["sources"][0]["document"] == "breach rules"

    @responses.activate
    def test_ask_with_groq(self, api, two_docs, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        responses.post(GROQ_URL, json={"choices": [{"message": {"content": "Within 72 hours [1]."}}]})
        body = api.post("/api/ask/", {"query": "When must a breach be notified?"}, format="json").json()
        assert body["mode"] == "llm:groq" and body["answer"] == "Within 72 hours [1]."
        sent = json.loads(responses.calls[0].request.body)
        assert "[1] (breach rules, page 1)" in sent["messages"][1]["content"]
        assert responses.calls[0].request.headers["Authorization"] == "Bearer test-key"

    @responses.activate
    def test_ask_with_gemini(self, api, two_docs, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        reply = {"candidates": [{"content": {"parts": [{"text": "Not later than 72 hours [1]."}]}}]}
        responses.post(GEMINI_URL, json=reply)
        body = api.post("/api/ask/", {"query": "When must a breach be notified?"}, format="json").json()
        assert body["mode"] == "llm:gemini" and body["answer"].startswith("Not later than 72 hours")

    @responses.activate
    def test_ask_falls_back_when_the_llm_fails(self, api, two_docs, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        responses.post(GROQ_URL, status=503)
        body = api.post("/api/ask/", {"query": "When must a breach be notified?"}, format="json").json()
        assert body["mode"] == "extractive" and "LLM call failed" in body["note"]

    def test_ask_with_no_documents(self, api):
        body = api.post("/api/ask/", {"query": "anything at all"}, format="json").json()
        assert body["mode"] == "none" and body["sources"] == []

    def test_index_is_rebuilt_from_the_database(self, api, two_docs):
        vectorstore.reset()  # like a new server process with an empty in-memory index
        res = api.post("/api/search/", {"query": "litter box and vaccinations"}, format="json").json()
        assert res["results"][0]["document_id"] == two_docs[1]["id"]
        assert vectorstore.count() == Chunk.objects.count()


def test_health(api, two_docs):
    body = api.get("/api/health/").json()
    assert body["status"] == "ok" and body["documents"] == 2
    assert body["chunks"] == body["indexed_chunks"] >= 2
    assert body["embedding_model"] == "wordllama-l2-supercat-256" and body["llm"] is None


def test_home_page(api):
    res = api.get("/")
    assert res.status_code == 200 and b"DRF RAG API" in res.content


def test_unknown_route_is_json_404(api):
    res = api.get("/api/nope/")
    assert res.status_code == 404


def test_chunk_size_setting_is_used(api, settings):
    settings.RAG = {**settings.RAG, "CHUNK_SIZE": 200, "CHUNK_OVERLAP": 30}
    body = upload(api, "long.txt", (BREACH_TEXT + " ").encode() * 4).json()
    assert body["chunk_count"] >= 4
