from django.test import TestCase
from django.core.exceptions import ValidationError
from django.core import management
from avocado.query import parsers
from ..models import Employee


class DataContextParserTestCase(TestCase):
    fixtures = ['query.json']

    def setUp(self):
        management.call_command('avocado', 'sync', 'query', quiet=True)

    def test_valid(self):
        # Single by id
        self.assertEqual(parsers.datacontext.validate({
            'id': 4,
            'operator': 'exact',
            'value': 'CEO'
        }, tree=Employee), None)

        # Single by dotted label
        self.assertEqual(parsers.datacontext.validate({
            'id': 'query.title.boss',
            'operator': 'exact',
            'value': 'CEO'
        }, tree=Employee), None)

        # Single by label list
        self.assertEqual(parsers.datacontext.validate({
            'id': ['query', 'title', 'boss'],
            'operator': 'exact',
            'value': 'CEO'
        }, tree=Employee), None)

        # Branch node
        self.assertEqual(parsers.datacontext.validate({
            'type': 'and',
            'children': [{
                'id': 4,
                'operator': 'exact',
                'value': 'CEO',
            }, {
                'id': 5,
                'operator': 'exact',
                'value': 'John',
            }]
        }, tree=Employee), None)

    def test_invalid(self):
        # Non-existent data field
        self.assertRaises(ValidationError, parsers.datacontext.validate, {
            'id': 99,
            'operator': 'exact',
            'value': 'CEO'
        }, tree=Employee)

        # Invalid structures
        # Object must be a dict
        self.assertRaises(ValidationError, parsers.datacontext.validate, [])

        # Object must be a dict
        self.assertRaises(ValidationError, parsers.datacontext.validate, None)

        # Invalid logical operator
        self.assertRaises(ValidationError, parsers.datacontext.validate, {'type': 'foo', 'children': []})

        # No children
        self.assertRaises(ValidationError, parsers.datacontext.validate, {'type': 'and', 'children': []})

        # 1 child
        self.assertRaises(ValidationError, parsers.datacontext.validate, {
            'type': 'and',
            'children': [{'id': 4, 'operator': 'exact', 'value': 'CEO'}]
        })

        # Missing 'value' key in first condition
        self.assertRaises(ValidationError, parsers.datacontext.validate, {
            'type': 'and',
            'children': [{
                'id': 4, 'operator': 'exact'
            }, {
                'id': 4, 'operator': 'exact', 'value': 'CEO'
            }]
        })

    def test_apply(self):
        node = parsers.datacontext.parse({
            'id': 4,
            'operator': 'exact',
            'value': True
        }, tree=Employee)

        self.assertEqual(str(node.apply().values('id').query), 'SELECT "query_employee"."id" FROM "query_employee" INNER JOIN "query_title" ON ("query_employee"."title_id" = "query_title"."id") WHERE ("query_title"."boss" = True  AND "query_title"."id" IS NOT NULL)')
        self.assertEqual(node.language, {'operator': 'exact', 'language': u'Boss is True', 'id': 4, 'value': True})

        # Branch node
        node = parsers.datacontext.parse({
            'type': 'and',
            'children': [{
                'id': 4,
                'operator': 'exact',
                'value': True,
            }, {
                'id': 5,
                'operator': 'exact',
                'value': 'John',
            }]
        }, tree=Employee)

        self.assertEqual(str(node.apply().values('id').query), 'SELECT "query_employee"."id" FROM "query_employee" INNER JOIN "query_title" ON ("query_employee"."title_id" = "query_title"."id") WHERE ("query_employee"."first_name" = John  AND "query_title"."boss" = True  AND "query_title"."id" IS NOT NULL)')
        self.assertEqual(node.language, {
            'type': 'and',
            'children': [{
                'id': 4,
                'operator': 'exact',
                'value': True,
                'language': 'Boss is True',
            }, {
                'id': 5,
                'operator': 'exact',
                'value': 'John',
                'language': 'First Name is equal to John',
            }]
        })


class DataViewParserTestCase(TestCase):
    fixtures = ['query.json']

    def setUp(self):
        management.call_command('avocado', 'sync', 'query', quiet=True)

    def test_valid(self):
        # Single by id
        self.assertEqual(parsers.datacontext.validate({
            'id': 4,
            'operator': 'exact',
            'value': 'CEO'
        }, tree=Employee), None)

        # Single by dotted label
        self.assertEqual(parsers.datacontext.validate({
            'id': 'query.title.boss',
            'operator': 'exact',
            'value': 'CEO'
        }, tree=Employee), None)

        # Single by label list
        self.assertEqual(parsers.datacontext.validate({
            'id': ['query', 'title', 'boss'],
            'operator': 'exact',
            'value': 'CEO'
        }, tree=Employee), None)

        # Branch node
        self.assertEqual(parsers.datacontext.validate({
            'type': 'and',
            'children': [{
                'id': 4,
                'operator': 'exact',
                'value': 'CEO',
            }, {
                'id': 5,
                'operator': 'exact',
                'value': 'John',
            }]
        }, tree=Employee), None)

    def test_invalid(self):
        # Non-existent data field
        self.assertRaises(ValidationError, parsers.datacontext.validate, {
            'id': 99,
            'operator': 'exact',
            'value': 'CEO'
        })

        # Invalid structures
        # Object must be a dict
        self.assertRaises(ValidationError, parsers.datacontext.validate, [])

        # Object must be a dict
        self.assertRaises(ValidationError, parsers.datacontext.validate, None)

        # Invalid logical operator
        self.assertRaises(ValidationError, parsers.datacontext.validate, {'type': 'foo', 'children': []})

        # No children
        self.assertRaises(ValidationError, parsers.datacontext.validate, {'type': 'and', 'children': []})

        # 1 child
        self.assertRaises(ValidationError, parsers.datacontext.validate, {
            'type': 'and',
            'children': [{'id': 4, 'operator': 'exact', 'value': 'CEO'}]
        })

        # Missing 'value' key in first condition
        self.assertRaises(ValidationError, parsers.datacontext.validate, {
            'type': 'and',
            'children': [{
                'id': 4, 'operator': 'exact'
            }, {
                'id': 4, 'operator': 'exact', 'value': 'CEO'
            }]
        })