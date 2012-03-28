from haystack.indexes import *
from haystack import site
from avocado.models import Concept

class ConceptIndex(RealTimeSearchIndex):
    text = CharField(document=True, use_template=True)

    def index_queryset(self):
        return Concept.objects.published()


site.register(Concept, ConceptIndex)