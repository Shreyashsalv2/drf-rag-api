from django.db.models import Count
from django.shortcuts import render
from rest_framework import mixins, permissions, status, viewsets
from rest_framework.decorators import action, api_view
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle

from rag.models import Chunk, Document
from rag.serializers import ChunkSerializer, DocumentSerializer, QuerySerializer, SourceSerializer, UploadSerializer
from rag.services import answer as answering
from rag.services import vectorstore
from rag.services.embeddings import MODEL_NAME
from rag.services.ingest import ingest_bytes, remove_document


class DocumentViewSet(
    mixins.ListModelMixin,
    mixins.RetrieveModelMixin,
    mixins.CreateModelMixin,
    mixins.DestroyModelMixin,
    viewsets.GenericViewSet,
):
    """Upload, list, inspect and delete documents. Uploads are throttled; deleting needs a token."""

    serializer_class = DocumentSerializer

    def get_queryset(self):
        return Document.objects.annotate(chunk_count=Count("chunks")).order_by("-created_at", "-id")

    def get_permissions(self):
        if self.action == "destroy":
            return [permissions.IsAuthenticated()]
        return [permissions.AllowAny()]

    def get_throttles(self):
        if self.action == "create":
            self.throttle_scope = "uploads"
            return [ScopedRateThrottle()]
        return []

    def create(self, request, *args, **kwargs):
        upload = UploadSerializer(data=request.data)
        upload.is_valid(raise_exception=True)
        file = upload.validated_data["file"]
        document = ingest_bytes(file.name, file.read(), title=upload.validated_data.get("title") or None)
        document = self.get_queryset().get(pk=document.pk)
        return Response(DocumentSerializer(document).data, status=status.HTTP_201_CREATED)

    def destroy(self, request, *args, **kwargs):
        remove_document(self.get_object())
        return Response(status=status.HTTP_204_NO_CONTENT)

    @action(detail=True, methods=["get"])
    def chunks(self, request, pk=None):
        page = self.paginate_queryset(Chunk.objects.filter(document_id=self.get_object().pk))
        return self.get_paginated_response(ChunkSerializer(page, many=True).data)


@api_view(["POST"])
def search(request):
    params = QuerySerializer(data=request.data)
    params.is_valid(raise_exception=True)
    data = params.validated_data
    hits = answering.search(data["query"], data["k"], data.get("document_ids"))
    return Response({"query": data["query"], "count": len(hits), "results": SourceSerializer(hits, many=True).data})


@api_view(["POST"])
def ask(request):
    params = QuerySerializer(data=request.data)
    params.is_valid(raise_exception=True)
    data = params.validated_data
    result = answering.answer(data["query"], data["k"], data.get("document_ids"))
    sources = SourceSerializer(result["sources"], many=True).data
    for n, source in enumerate(sources, start=1):
        source["n"] = n
    return Response(
        {
            "question": data["query"],
            "answer": result["answer"],
            "mode": result["mode"],
            "note": result["note"],
            "sources": sources,
        }
    )


@api_view(["GET"])
def health(request):
    vectorstore.ensure_synced()
    return Response(
        {
            "status": "ok",
            "documents": Document.objects.count(),
            "chunks": Chunk.objects.count(),
            "indexed_chunks": vectorstore.count(),
            "embedding_model": MODEL_NAME,
            "llm": answering.llm_provider(),
        }
    )


def home(request):
    return render(request, "rag/index.html")
