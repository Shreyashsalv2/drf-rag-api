from django.conf import settings
from rest_framework import serializers

from rag.models import Chunk, Document
from rag.services.extract import ExtractionError, file_type_for


class DocumentSerializer(serializers.ModelSerializer):
    chunk_count = serializers.IntegerField(read_only=True)

    class Meta:
        model = Document
        fields = [
            "id",
            "title",
            "filename",
            "file_type",
            "size_bytes",
            "pages",
            "is_sample",
            "chunk_count",
            "created_at",
        ]
        read_only_fields = fields


class UploadSerializer(serializers.Serializer):
    file = serializers.FileField()
    title = serializers.CharField(max_length=255, required=False, allow_blank=True)

    def validate_file(self, upload):
        try:
            file_type_for(upload.name)
        except ExtractionError as error:
            raise serializers.ValidationError(str(error)) from error
        limit = settings.RAG["MAX_UPLOAD_MB"] * 1024 * 1024
        if upload.size > limit:
            raise serializers.ValidationError(f"File is larger than {settings.RAG['MAX_UPLOAD_MB']:g} MB.")
        if upload.size == 0:
            raise serializers.ValidationError("File is empty.")
        return upload


class ChunkSerializer(serializers.ModelSerializer):
    class Meta:
        model = Chunk
        fields = ["id", "index", "page", "start_char", "text"]


class QuerySerializer(serializers.Serializer):
    query = serializers.CharField(min_length=2, max_length=500, trim_whitespace=True)
    k = serializers.IntegerField(min_value=1, max_value=20, default=5)
    document_ids = serializers.ListField(
        child=serializers.IntegerField(min_value=1), required=False, allow_empty=False, max_length=50
    )


class SourceSerializer(serializers.Serializer):
    chunk_id = serializers.IntegerField(source="chunk.id")
    document_id = serializers.IntegerField(source="chunk.document_id")
    document = serializers.CharField(source="chunk.document.title")
    page = serializers.IntegerField(source="chunk.page")
    score = serializers.FloatField()
    text = serializers.CharField(source="chunk.text")
