import re
from warnings import warn
from datetime import datetime
from math import ceil, floor, pow, fabs

from django import forms
from django.db import models
from django.db.models import Count, Min, Max
from django.contrib.auth.models import Group
from django.contrib.sites.models import Site
from django.db.models.fields import FieldDoesNotExist

from avocado.modeltree import DEFAULT_MODELTREE_ALIAS, trees
from avocado.fields.translate import library
from avocado.fields.managers import FieldManager
from avocado.fields import mixins

__all__ = ('Field',)

COERCED_DATATYPES = (
    (re.compile(r'^integer|float|decimal|big|positive|small|auto'), 'number'),
    (re.compile(r'^char|text|file|ipaddress|slug'), 'string'),
)

REVIEW_CHOICES = (
    ('Unreviewed', 'Unreviewed'),
    ('Embargoed', 'Embargoed'),
    ('Deprecated', 'Deprecated'),
    ('Waiting', 'Waiting'),
    ('Curation Required', 'Curation Required'),
    ('Curation Pending', 'Curation Pending'),
    ('Integration Required', 'Integration Required'),
    ('Integration Pending', 'Integration Pending'),
    ('Finalized', 'Finalized'),
)

class Field(mixins.Mixin):
    """The `Field' class stores off meta data about a "field of
    interest" located on another model. This, in a sense, provides a way to
    specify the fields that can be utilized by the query engine.

    There are three cases in which a "field of interest" should be stored:

        defintion/vocabulary - at the very least, storing off a field provides
        the ability specify a `description' and `keywords' (or aliases) associated
        with the field.

        queryability - the field can be associated with one or more
        `Criterions'. at minimum, this provides the ability to query by
        this field. in a more complex scenario, the field can act as a dependent
        or dependency on other fields.

        reporting - the field can be associated with one or more `Columns'
        which allows for generating reports (results) in-browser or exporting to
        another format.
    """
    app_name = models.CharField(max_length=100)
    model_name = models.CharField(max_length=100)
    field_name = models.CharField(max_length=100)

    name = models.CharField(max_length=100)
    description = models.TextField(null=True, blank=True)
    keywords = models.CharField(max_length=100, null=True, blank=True)

    # the ``search_doc`` refers to the aggregated text that can be fulltext
    # searched. if the database backend supports it, this will also be used
    # by the fulltext indexing to construct the search index
    search_doc = models.TextField(null=True, db_index=True, editable=False)

    status = models.CharField('review status', max_length=40,
        choices=REVIEW_CHOICES, blank=True, null=True)
    note = models.TextField('review note', null=True)
    reviewed = models.DateTimeField('last reviewed', null=True)

    is_public = models.BooleanField(default=False)
    group = models.ForeignKey(Group, null=True, blank=True)
    sites = models.ManyToManyField(Site, blank=True)

    translator = models.CharField(max_length=100, choices=library.choices(),
        blank=True, null=True)

    # specify a constrained list of choices for this field, applies for
    # valiation and for display in the UI
    enable_choices = models.BooleanField(default=False)
    choices_handler = models.TextField(null=True, blank=True, help_text="""
        Allowed callbacks include specifying:
            1. a constant name on the model
            2. a constant name on the model's module
            3. a string that can be evaluated
    """)

    objects = FieldManager()

    class Meta(object):
        app_label = 'avocado'
        unique_together = ('app_name', 'model_name', 'field_name')
        ordering = ('name',)

    def __unicode__(self):
        if self.name:
            name = '%s [%s]' % (self.name, self.model_name)
        else:
            name = '.'.join([self.app_name, self.model_name, self.field_name])
        return u'%s' % name

    def save(self):
        self.reviewed = datetime.now()
        super(Field, self).save()

    def natural_key(self):
        return (self.app_name, self.model_name, self.field_name)

    def _get_module(self):
        "Helper for determining ``choices``, if ``choices_handler`` is defined."
        if not hasattr(self, '_module'):
            self._module = __import__(self.model.__module__)
        return self._module
    module = property(_get_module)

    def _get_model(self, app_name=None, model_name=None):
        "Returns the model class this field is associated with."
        if not hasattr(self, '_model') or (app_name and model_name):
            app_name = app_name or self.app_name
            model_name = model_name or self.model_name
            self._model = models.get_model(app_name, model_name)
        return self._model
    model = property(_get_model)

    def _get_field(self, field_name=None):
        "Returns the field object from model."
        if not hasattr(self, '_field') or field_name:
            field_name = field_name or self.field_name
            try:
                self._field = self.model._meta.get_field_by_name(field_name)[0]
            except FieldDoesNotExist:
                self._field = None
        return self._field
    field = property(_get_field)

    def _get_datatype(self):
        if not hasattr(self, '_datatype'):
            internal = self.field.get_internal_type().lower()
            for p, t in COERCED_DATATYPES:
                if p.match(internal):
                    self._datatype = t
                    break
            else:
                if internal.endswith('field'):
                    internal = internal[:-5]
                self._datatype = internal
        return self._datatype
    datatype = property(_get_datatype)

    def _get_choices(self, choices_handler=None):
        """Returns a distinct set of choices for this field.

        If a ``choices_handler`` is not defined, a distinct list of values are
        retrieved from the database.

        If ``choices_handler`` is defined, the handler is passed through a series
        of evaluators to try and convert it into a native object.
        """
        if not hasattr(self, '_choices') or choices_handler:
            choices = None

            # override boolean type fields
            if self.datatype in ('boolean', 'nullboolean'):
                choices = [(True, 'Yes'), (False, 'No')]
                if self.datatype == 'nullboolean':
                    choices.append((None, 'No Data'))
                choices = tuple(choices)

            elif self.enable_choices:
                choices_handler = choices_handler or self.choices_handler

                # use introspection
                if not choices_handler:
                    name = self.field_name
                    choices = list(self.model.objects.values_list(name,
                        flat=True).order_by(name).distinct())
                    choices = zip(choices, map(lambda x: x is None and 'No Data' or str(x), choices))

                # attempt to evaluate custom handler
                else:
                    from avocado.fields import evaluators
                    choices = evaluators.evaluate(self)

            self._choices = choices
        return self._choices
    choices = property(_get_choices)

    def _get_distinct_choices(self):
        "Returns a list of the raw values."
        warn('Use Field.values instead', DeprecationWarning)
        return self.values
    distinct_choices = property(_get_distinct_choices)

    def _get_values(self):
        "Returns a list of the values."
        if self.choices is not None:
            return tuple(map(lambda x: x[0], self.choices))
        return ()
    values = property(_get_values)

    def distribution(self, exclude=[], min_count=None, max_points=100,
        order_by='field', smooth=0.01, annotate_by='id', **filters):

        """Builds a GROUP BY queryset for use as a value distribution.

        ``exclude`` - a list of values to be excluded from the distribution. it
        may be desired to exclude NULL values or the empty string.

        .. note::

            the default behavior of the ``exclude`` argument is to do a SQL
            equivalent of NOT IN (...). if ``None`` is included, it will have
            a custom behavior of IS NOT NULL and will be removed from the IN
            clause. default is to include all values

        ``max_points`` - the maximum number of points to be include in the
        distribution. the min and max values are always included, then a random
        sample is taken from the distribution. default is 100

        ``order_by`` - specify an ordering for the distribution. the choices are
        'count', 'field', or None. default is 'count'

        ``filters`` - a dict of filters to be applied to the queryset before
        the count annotation.
        """
        name = str(self.field_name)

        # get base queryset
        dist = self.model.objects.values(name)

        # exclude certain values (e.g. None, '')
        if exclude:
            exclude = set(exclude)
            kwargs = {}

            # special case for null values
            if None in exclude:
                kwargs['%s__isnull' % name] = True
                exclude.remove(None)

            kwargs['%s__in' % name] = exclude
            dist = dist.exclude(**kwargs)

        # apply filters before annotation is made
        if filters:
            dist = dist.filter(**filters)

        
#############################################################
######### NEW STUFF HERE ####################################
#############################################################
        if self.datatype == 'number' and smooth >= 0:
            # get data from queryset
            ncount = self.model.objects.aggregate(**{'nn':Count(name),
                'min':Min(name), 'max':Max(name)})
            n = ncount['nn']
            minx = ncount['min']
            maxx = ncount['max']

            # evaluate
            dist = dist.values_list(name, flat=True)
            
            # raw ordered data
            dist = dist.order_by(name)


            # if n is over 1 million use simpler functon to get 
            # bin width. REVISIT THIS... 
            # Change this so that when n is too large
            # data isn't stored in memory.
            if n >= 1000000:
                max_min = maxx-minx
                h = float(maxx)/ceil(pow(n, (1.0/2.0)))
            else:
                q1 = dist[int(ceil(n*.25))]
                q3 = dist[int(floor(n*.75))]
                iqr = q3-q1
                h = 2 * float(iqr) * pow(n, -(1.0/3.0))
            print "n:{0} h:{1} iqr:{2}".format(n, h, iqr) 
            bin_data = []
            bin = float(minx) + h
            bin_height = 0

            from time import clock

            t1 = clock()
            while bin <= maxx:
                filt_range = {name+'__lte': bin, name+'__gt': bin-h}
                cnt = dist.filter(**filt_range)#.aggregate(**{'y':Count(name)})
                if cnt: 
                    x = bin
                    y = len(cnt)
                    prev = (0,0)
                    # print x,y
                    if bin_data:
                        prev = bin_data.pop()

                    # compare current bin to previous 
                    # if prev bin is too small, the current
                    # bin takes in previous.
                    # Previous bin takes in current bin, 
                    # if current bin is too small
                    if (y*smooth) > prev[1]:
                        fact = prev[1]/y
                        bin_data.append(((x-(h/2)-fact), y+prev[1]))
                    elif prev[1]*smooth > y:
                            fact = y/prev[1]
                            bin_data.append((prev[0]+fact, y+prev[1]))
                    else:
                        bin_data.append(prev)
                        bin_data.append((x-(h/2), y))                   
                bin = bin + h 
            print clock()-t1

            # Set up an empty list, the first bin, and
            # initate bin_height to 0
#            t1 = clock()
#            for data_pt in dist:
#                data_pt = float(data_pt)
#                # If data point is less than the bin
#                # add 1 to the bin height
#                if data_pt <= bin:
#                    bin_height += 1
#                if data_pt > bin or data_pt == maxx :
#                    x = bin
#                    y = bin_height
#                    prev = (0,0)
#                    if bin_data:
#                        prev = bin_data.pop()
#
#                    # compare current bin to previous 
#                    # if prev bin is too small, the current
#                    # bin takes in previous.
#                    # Previous bin takes in current bin, 
#                    # if current bin is too small
#                    if (y*smooth) > prev[1]:
#                        fact = prev[1]/y
#                        bin_data.append(((x-(h/2)-fact), y+prev[1]))
#                    elif prev[1]*smooth > y:
#                        fact = y/prev[1]
#                        bin_data.append((prev[0]+fact, y+prev[1]))
#                    else:
#                        bin_data.append(prev)
#                        bin_data.append((x-(h/2), y))                   
#                    bin_height = 0
#                    if data_pt != maxx:
#                        # increment to next bin until data_pt
#                        # is within the bin. Add 1 to height and 
#                        # move to next data_pt
#                        while data_pt > bin:
#                            bin += h
#                        bin_height += 1
#            print clock()-t1 

            dist = bin_data
        else:   
            # apply annotation
            dist = dist.annotate(count=Count(annotate_by))

            # evaluate
            dist = dist.values_list(name, 'count')
            
                        
            # apply ordering
            if order_by == 'count':
                dist = dist.order_by('count')
            elif order_by == 'field':
                dist = dist.order_by(name)

        if len(dist) < 3:
            return tuple(dist)

        return tuple(dist)

    def query_string(self, operator=None, using=DEFAULT_MODELTREE_ALIAS):
        "Returns a django lookup string relative to the ``modeltree`` object."
        modeltree = trees[using]
        nodes = modeltree.path_to(self.model)
        return modeltree.query_string(nodes, self.field_name, operator)

    def order_string(self, direction='asc', using=DEFAULT_MODELTREE_ALIAS):
        qs = self.query_string(using=using)
        if direction.lower() == 'desc':
            return '-' + qs
        return qs

    def translate(self, operator=None, value=None, using=DEFAULT_MODELTREE_ALIAS, **context):
        trans = library.get(self.translator)
        return trans(self, operator, value, using, **context)

    def query_by_value(self, operator, value, using=DEFAULT_MODELTREE_ALIAS):
        modeltree = trees[using]
        meta = self.translate(operator, value, using)
        queryset = modeltree.root_model.objects.all()

        if meta['annotations']:
            queryset = queryset.annotate(**meta['annotations'])
        if meta['condition']:
            queryset = queryset.filter(meta['condition'])
        return queryset

    def formfield(self, formfield=None, widget=None, **kwargs):
        "Returns the default `formfield' instance for the `field' type."
        if formfield is None:
            formfield = self.field.formfield

        if 'label' in kwargs:
            label = kwargs.pop('label')
        else:
            label = self.name.title()

        if not widget and self.enable_choices:
            widget = forms.SelectMultiple(choices=self.choices)

        return formfield(label=label, widget=widget, **kwargs)

