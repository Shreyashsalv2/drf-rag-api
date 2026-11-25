from django.urls import include, path
from rest_framework.routers import DefaultRouter

from rag import views

router = DefaultRouter(trailing_slash="/?")
router.register("documents", views.DocumentViewSet, basename="document")

urlpatterns = [
    path("", views.home, name="home"),
    path("api/", include(router.urls)),
    path("api/search/", views.search, name="search"),
    path("api/ask/", views.ask, name="ask"),
    path("api/health/", views.health, name="health"),
]
