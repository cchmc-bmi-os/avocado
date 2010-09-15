from avocado.columns import cache
from avocado.modeltree import DEFAULT_MODELTREE_ALIAS

def get_rules(columns, ftype):
    return map(lambda x: x.rule(ftype), columns)

def add_columns(queryset, columns, using=DEFAULT_MODELTREE_ALIAS):
    """Takes a `queryset' and ensures the proper table join have been
    setup to display the table columns.
    """
    # TODO determine if the aliases can contain more than one reference
    # to a table
    
    rules = []
    # add queryset model's pk field
    aliases = [(queryset.model._meta.db_table,
        queryset.model._meta.pk.column)]

    for pk in columns:
        column = cache.get(pk)
        queryset, _aliases = column.add_fields_to_queryset(queryset, using)
        rules.append(columns.rule)
        aliases.extend(_aliases)

    queryset.query.select = aliases
    return queryset, rules

def add_ordering(queryset, column_orders, using=DEFAULT_MODELTREE_ALIAS):
    """Applies column ordering to a queryset. Resolves a Column's
    fields and generates the `order_by' paths.
    """
    queryset.query.clear_ordering()
    orders = []

    for pk, direction in column_orders:
        column = cache.get(pk)
        _orders = column.get_ordering_for_queryset(direction, using)
        orders.extend(_orders)

    return queryset.order_by(*orders)
