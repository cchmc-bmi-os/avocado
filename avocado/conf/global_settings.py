# For custom translators, this is the name of the module that will be 
# introspected per app for registered translators
TRANSLATOR_MODULE_NAME = 'translate'

# Enables a many-to-many relationship between ``Field`` and ``Group`` to allow
# for user-specific authorization to certain fields.
FIELD_GROUP_PERMISSIONS =  False

# For custom formatters, this is the name of the module that will be
# introspected per app for registered formatters
FORMATTER_MODULE_NAME = 'format'

# A definition for customizing the ``Column`` model. Each key in the dict will
# be the name of the column formatter type as well as the name of the model
# field name (with FORMATTER_FIELD_SUFFIX). These correspond to the output
# formats produced by the column formatter library
FORMATTER_TYPES = {}

# The formatter type field suffix, e.g. given a formatter type 'csv', the
# field that will be added to the ``Column`` model will be called 'csv_fmt'
FORMATTER_FIELD_SUFFIX = '_fmtr'

# A tuple of ``Column`` ids that represents the default ordering for reports
COLUMN_ORDERING = ()

# A tuple of ``Column`` ids that represents the default columns to be shown
# in a report
COLUMNS = ()

# For custom viewsets, this is the name of the module that will be
# introspected per app for registered viewsets
VIEWSET_MODULE_NAME = 'viewset'

# A dict of modeltree configurations. Each config should contrain the necessary
# keyword args for constructing a default ``ModelTree`` instance. View the 
# ``ModelTree`` docs for potential arguments
MODELTREES = {}