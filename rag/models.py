from django.db import models


class Document(models.Model):
    """An uploaded file. Its text lives in Chunk rows; ChromaDB holds the chunk embeddings."""

    class FileType(models.TextChoices):
        PDF = "pdf", "PDF"
        DOCX = "docx", "Word"
        TXT = "txt", "Text"
        MD = "md", "Markdown"

    title = models.CharField(max_length=255)
    filename = models.CharField(max_length=255)
    file_type = models.CharField(max_length=8, choices=FileType.choices)
    size_bytes = models.PositiveIntegerField()
    pages = models.PositiveIntegerField(default=1)
    is_sample = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at", "-id"]

    def __str__(self) -> str:
        return self.title


class Chunk(models.Model):
    """A piece of a document's text, small enough to embed and to quote back as a source."""

    document = models.ForeignKey(Document, related_name="chunks", on_delete=models.CASCADE)
    index = models.PositiveIntegerField()
    page = models.PositiveIntegerField()
    start_char = models.PositiveIntegerField()
    text = models.TextField()

    class Meta:
        ordering = ["document_id", "index"]
        constraints = [models.UniqueConstraint(fields=["document", "index"], name="unique_chunk_index")]

    @property
    def vector_id(self) -> str:
        return f"chunk-{self.pk}"
