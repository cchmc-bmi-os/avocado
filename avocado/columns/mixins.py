from django.db import models
from django.utils.importlib import import_module

from avocado.conf import settings
from avocado.utils.mixins import create_mixin
from avocado.columns.format import library

# attempt to import the user-defined mixin class if specified
if settings.FIELD_MIXIN_PATH is not None:
    path = settings.FIELD_MIXIN_PATH.split('.')
    mixin_name = path.pop(-1)
    mod = import_module('.'.join(path))
    Mixin = getattr(mod, mixin_name)
else:
    Mixin = models.Model

fields = {}
for name in settings.FORMATTER_TYPES:
    fn = name + settings.FORMATTER_FIELD_SUFFIX
    fields[fn] = models.CharField('%s formatter' % name, max_length=100,
        choices=library.choices(name), blank=True)

Mixin = create_mixin('Mixin', __name__, bases=(Mixin,), fields=fields)
