from django.utils.importlib import import_module
from modeltree.tree import trees, ModelTree
from modeltree.utils import M, print_traversal_tree
from avocado.formatters import RawFormatter
from avocado.conf import settings
from avocado.models import DataField
from .operators_long import LONG_OPERATORS
from django.db.models import Max, Min, Avg
from django.db.models import F
from ibemc.models import Subject
from ibemc.models_longitudinal import VisitLongitudinal

QUERY_PROCESSOR_DEFAULT_ALIAS = 'default'
LONGITUDINAL=1
NONLONGITUDINAL=-1
SUBJECT_TO_TIME_AXIS = 'record__visit_longitudinal__long_index'

class QueryProcessor(object):
    """Prepares and builds a QuerySet for export.

    Overriding or extending these methods enable customizing the behavior
    pre/post-construction of the query.
    """
    def __init__(self, context=None, view=None, tree=None, include_pk=True):
        self.context = context
        self.view = view
        self.tree = tree
        self.include_pk = include_pk

    def get_queryset(self, queryset=None, **kwargs):
        "Returns a queryset based on the context and view."
        if self.context:
            queryset = \
                self.context.apply(queryset=queryset, tree=self.tree)

        if self.view:
            queryset = self.view.apply(queryset=queryset, tree=self.tree,
                                       include_pk=self.include_pk)

        ## Returns all subjects
        if queryset is None:
            queryset = trees[self.tree].get_queryset()

        return queryset

    def get_exporter(self, klass, **kwargs):
        "Returns an exporter prepared for the queryset."
        exporter = klass(self.view)

        if self.include_pk:
            pk_name = trees[self.tree].root_model._meta.pk.name
            exporter.add_formatter(RawFormatter(keys=[pk_name]), index=0)

        return exporter

    def get_iterable(self, offset=None, limit=None, queryset=None, **kwargs):
        "Returns an iterable that can be used by an exporter."
        if queryset is None:
            queryset = self.get_queryset(**kwargs)

        if offset is not None and limit is not None:
            queryset = queryset[offset:offset + limit]
        elif offset is not None:
            queryset = queryset[offset:]
        elif limit is not None:
            queryset = queryset[:limit]

        compiler = queryset.query.get_compiler(queryset.db)
        return compiler.results_iter()



## Or add a get_longitudinal_queryset(..., events_context)
class LongitudinalQueryProcessor(QueryProcessor):
    def __init__(self, context=None, view=None, tree=None, include_pk=True, long_events=None, study_models=None):
        super(LongitudinalQueryProcessor, self).__init__(context, view, tree, include_pk)
        self.long_events = long_events
        self.study_models = study_models

    def record_subject(self, node):
        record = self.study_models['record']
        subject = self.study_models['subject']
        if node.model_name==record:
            return LONGITUDINAL

        if node.model_name==subject:
            return NONLONGITUDINAL

        return None

    def is_longitudinal(self, klass):
        """Check if visits is between the concept and subject.

        The code find which of Record or Subject is closest to the model to
        identify the longitudinality.

        """

        assert type(klass) != DataField, 'Parameter is not an avocado.DataField, it should be a model'
        mt = trees.create(klass)

        node = mt.root_node
        found = self.record_subject(node)
        if found:
            return found == LONGITUDINAL

        children = node.children
        while children:
            next_children = []
            for n in children:
                found = self.record_subject(n)
                if found:
                    return found == LONGITUDINAL
                next_children += n.children

            children = next_children


    def get_queryset(self, queryset=None, **kwargs):
        "Returns a queryset based on the context and view and longitudinal parameters."
        if self.context:
            queryset = \
                self.context.apply(queryset=queryset, tree=self.tree)

        if self.view:
            queryset = self.view.apply(queryset=queryset, tree=self.tree,
                                       include_pk=self.include_pk)

        event_timeline = None
        if self.long_events:
            if 'instance' in kwargs:
                model = kwargs['instance']
            else:
                model = queryset.model

            ## get the Subjects matching the context
            long_tree = trees.create(Subject)

            p = QueryProcessor(tree=long_tree, context=self.context, include_pk=True)
            long_qs = p.get_queryset()

            ## Limits the records to those matching the events
            long_records = self.long_events.apply(queryset=long_qs, tree=long_tree)

            ## Identify the timeline of the event i.e. values of longitudinal indexes when event is true
            event_timeline = long_records.annotate(Min('record__visit_longitudinal__long_index')).annotate(Max('record__visit_longitudinal__long_index'))

        return event_timeline

    def get_long_stat(self, visits, param, model, field):
        visit_filter = M(visits, **param)
        visit = VisitLongitudinal.objects.filter(visit_filter)

        if model == 'VisitLongitudinal':
            q_str = ''
        else:
            q_str = ModelTree.query_string(visits, model) + '__'

        q_str = q_str + field
        try:
            avg_ope = visit.aggregate(Avg(q_str)).values()[0]
        except:
            avg_ope = None
        return avg_ope


    def get_long_stats(self, timeline, long_events, model, field):
        visits = trees.create(VisitLongitudinal)
        # print_traversal_tree(visits)

        results = {}
        for p in timeline:
            pat_id = p.id
            index_min = p.record__visit_longitudinal__long_index__min
            index_max = p.record__visit_longitudinal__long_index__max

            # subjects_filter = M(subjects, id=pat_id, record__visitlongitudinal__longitudinal_study_index__lt=index_min)
            # subjects_filter = M(subjects, id=pat_id, record__visit_longitudinal__id__gte=index_min)
            # pat = Subject.objects.filter(subjects_filter)

            param = {'longitudinal_study_index__gt': index_max,
                     'record__subject': pat_id}
            after = self.get_long_stat(visits, param, model, field)

            param = {'longitudinal_study_index__lt': index_min,
                     'record__subject': pat_id}
            before = self.get_long_stat(visits, param, model, field)

            results[p.id] = {'after': after,
                             'before': before}

        before = [ results[p]['before'] for p in results if results[p]['before'] ]
        after = [ results[p]['after'] for p in results if results[p]['after'] ]

        b = sum(before)/max(len(before),1)
        a = sum(after)/max(len(after),1)

        return {'before': b, 'after':a}


class QueryProcessors(object):
    def __init__(self, processors):
        self.processors = processors
        self._processors = {}

    def __getitem__(self, key):
        return self._get(key)

    def __len__(self):
        return len(self._processors)

    def __nonzero__(self):
        return True

    def _get(self, key):
        # Import class if not cached
        if key not in self._processors:
            toks = self.processors[key].split('.')
            klass_name = toks.pop()
            path = '.'.join(toks)
            klass = getattr(import_module(path), klass_name)
            self._processors[key] = klass
        return self._processors[key]

    def __iter__(self):
        return iter(self.processors)

    @property
    def default(self):
        return self[QUERY_PROCESSOR_DEFAULT_ALIAS]


query_processors = QueryProcessors(settings.QUERY_PROCESSORS)
