# DRF RAG API

Upload PDF, DOCX or text files and ask questions about them. The answers cite the passage and page they came from. Built with Django REST Framework and ChromaDB.

**Live demo:** https://drf-rag-api.vercel.app (the first request after a pause takes a few seconds while the sample documents load)

![Asking the sample documents how quickly a data breach must be reported](docs/screenshot.png)

## What it does

1. **Upload** a PDF, DOCX, TXT or Markdown file (`POST /api/documents/`, multipart).
2. **Extract** the text: one entry per page for PDFs (pypdf), and paragraphs plus tables for DOCX (python-docx). The text is then cleaned: hyphenated words are rejoined and whitespace is normalised.
3. **Chunk** each page into overlapping pieces, 1,200 characters with 180 characters of overlap by default.
   - The splitter breaks at the largest natural boundary it can find: a blank line, then a line, then a sentence, then a word.
   - Chunks never cross a page, and each one keeps its page number and character offset.
4. **Embed** each chunk with **WordLlama**, a 256-dimension embedding model whose weights ship inside its pip package, so no API key or model download is needed. The vectors are stored in **ChromaDB** with cosine distance.
5. **Search** (`POST /api/search/`) returns the k most similar chunks, each with its document, page and score. You can limit the search to particular documents.
6. **Ask** (`POST /api/ask/`) retrieves chunks the same way and then answers:
   - **No API key set:** an *extractive* answer, made of the three sentences closest to the question, each tagged with its source number.
   - **`GROQ_API_KEY` or `GEMINI_API_KEY` set:** an LLM writes the answer from the numbered sources. It is told to cite them and to say when they don't cover the question.
   - **LLM call fails or times out:** the API falls back to the extractive answer and says so.

The database (`Document` and `Chunk` rows) is the source of truth, and ChromaDB is an index over it. If the index is empty or out of step (a new process, or an in-memory index), it is rebuilt from the table.

## Retrieval evaluation

`samples/` holds 5 documents in 3 formats (PDF, DOCX, Markdown). They are built from 58 articles of the GDPR by `scripts/make_samples.py`. `samples/questions.json` has 25 hand-written questions, each labelled with the article(s) that answer it.

`python manage.py evaluate_rag` chunks the documents at 3 sizes and checks whether a chunk from the right article comes back in the top 5:

| Chunk size / overlap | Chunks | Recall@5 | MRR |
|---|---|---|---|
| 400 / 60 | 410 | 22 / 25 (0.88) | 0.698 |
| 800 / 120 | 201 | 22 / 25 (0.88) | 0.661 |
| **1,200 / 180** | **136** | **23 / 25 (0.92)** | **0.737** |

That table is why 1,200 characters is the default. CI re-runs the evaluation and fails if recall@5 drops below 0.88. Full per-question results are in [`results/eval.json`](results/eval.json).

Some honest notes on these numbers:
- WordLlama is a static embedding model. It handles synonyms that keyword search misses: "delete my data" finds Article 17, on erasure, at rank 3. But it is weaker than a full transformer model.
- The two questions it misses at 1,200 characters are "sensitive health data" and "correcting inaccurate personal data".

## API

| Method | Path | Notes |
|---|---|---|
| GET | `/api/health/` | counts of documents, chunks and indexed chunks; embedding model; LLM provider in use |
| GET | `/api/documents/` | paginated list with page and chunk counts |
| POST | `/api/documents/` | multipart `file` (+ optional `title`). Limited to 4 MB and throttled to 10 uploads/hour per client |
| GET | `/api/documents/{id}/` | one document |
| GET | `/api/documents/{id}/chunks/` | its chunks with page and offset (paginated) |
| DELETE | `/api/documents/{id}/` | needs `Authorization: Token <key>` (DRF token auth); removes the rows and the vectors |
| POST | `/api/search/` | `{"query": "...", "k": 5, "document_ids": [1, 2]}` |
| POST | `/api/ask/` | same body; returns `answer`, `mode` (`extractive`, `llm:groq` or `llm:gemini`) and numbered `sources` |

Every error has the same JSON shape: `{"error": {"code": "invalid_request", "message": "...", "details": {...}}}`. The status codes are:
- 400: invalid input, unsupported or unreadable file
- 401: delete without a token
- 404: unknown document
- 429: too many uploads
- 500: unexpected error, with no internals leaked

## Stack

Python 3.12 · Django 5.2 · Django REST Framework · ChromaDB · WordLlama · pypdf · python-docx · PostgreSQL 16 (SQLite for the demo) · pytest + pytest-django · Docker Compose · GitHub Actions · Vercel

## Run it

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
python manage.py migrate && python manage.py load_samples
python manage.py runserver            # http://localhost:8000
```

With PostgreSQL and a persistent ChromaDB index:

```bash
docker compose up --build             # API on http://localhost:8000, PostgreSQL on :5432
```

Optional settings (environment variables): `GROQ_API_KEY` / `GROQ_MODEL`, `GEMINI_API_KEY` / `GEMINI_MODEL`, `DATABASE_URL`, `CHROMA_PATH`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `MAX_UPLOAD_MB`, `UPLOAD_RATE`.

To create a token for deletes: `python manage.py createsuperuser`, then `python manage.py drf_create_token <username>`.

## Tests

```bash
pytest                                              # SQLite
DATABASE_URL=postgres://postgres:postgres@localhost:5432/rag pytest
ruff check . && ruff format --check .
python manage.py evaluate_rag --min-recall 0.88
```

There are 40 tests. They cover:
- **Extraction:** PDF pages, DOCX tables, broken and image-only files.
- **Chunking:** size limits, offsets that point back to the page, overlap, page boundaries.
- **Every endpoint:** upload validation, size limit, throttling, token-protected delete removing the vectors, search ranking and filters, input validation.
- **Answers:** the extractive answer with citations; Groq and Gemini answers with the HTTP calls mocked; the fallback when the LLM fails; rebuilding the index from the database.
- **Evaluation:** a retrieval-quality check on the samples.

GitHub Actions runs two jobs:
1. Lint, the migration check, the tests on **PostgreSQL 16 and on SQLite**, and the retrieval evaluation.
2. Builds the Docker Compose stack, then uploads, searches and asks on the running container.

## Demo limits

The Vercel demo uses SQLite and an in-memory index in `/tmp`, and loads the samples on each cold start. Uploads work, but they disappear when the instance restarts and are not shared between instances. Docker Compose keeps everything in PostgreSQL and a persistent ChromaDB volume.

## Data and license

- **Sample documents:** built from the [Structured GDPR JSON dataset](https://github.com/gbasilveira/GDPR-JSON) by Guilherme Silveira (CC BY 4.0). See [`data/README.md`](data/README.md).
- **Code:** MIT. See [LICENSE](LICENSE).

This is a demo, not legal advice.
