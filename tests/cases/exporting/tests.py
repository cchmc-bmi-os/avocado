import os
from django.test import TestCase
from django.http import HttpResponse
from django.template import Template
from django.core import management
from avocado import export
from avocado.formatters import RawFormatter
from avocado.models import DataField, DataConcept, DataConceptField, DataView
from . import models


class ExportTestCase(TestCase):
    fixtures = ['export.json']

    def setUp(self):
        management.call_command('avocado', 'init', 'exporting', quiet=True)

    def test_view(self):
        salary_field = DataField.objects.get_by_natural_key('exporting', 'title', 'salary')
        salary_concept = DataConcept()
        salary_concept.save()
        DataConceptField(concept=salary_concept, field=salary_field, order=1).save()

        view = DataView(json={'ordering': [[salary_concept.pk, 'desc']]})
        query = view.apply(tree=models.Employee).raw()

        # Ick..
        exporter = export.CSVExporter(query, view)
        exporter._format.params.insert(0, (RawFormatter(keys=['pk']), 1))
        exporter._format.row_length += 1

        buff = exporter.write()
        buff.seek(0)

        lines = buff.read().splitlines()
        # Skip the header
        self.assertEqual([int(x) for x in lines[1:]], [2, 4, 6, 1, 3, 5])


class FileExportTestCase(TestCase):
    fixtures = ['export.json']

    def setUp(self):
        management.call_command('avocado', 'init', 'exporting', quiet=True)
        first_name_field = DataField.objects.get_by_natural_key('exporting', 'employee', 'first_name')
        first_name_field.description = 'First Name'
        last_name_field = DataField.objects.get_by_natural_key('exporting', 'employee', 'last_name')
        last_name_field.description = 'Last Name'
        title_field = DataField.objects.get_by_natural_key('exporting', 'title', 'name')
        title_field.description = 'Employee Title'
        salary_field = DataField.objects.get_by_natural_key('exporting', 'title', 'salary')
        salary_field.description = 'Salary'
        is_manager_field = DataField.objects.get_by_natural_key('exporting', 'employee', 'is_manager')
        is_manager_field.description = 'Is a Manager?'

        [x.save() for x in [first_name_field, last_name_field, title_field,
            salary_field, is_manager_field]]

        employee_concept = DataConcept()
        employee_concept.name = 'Employee'
        employee_concept.description = 'A Single Employee'
        employee_concept.save()

        DataConceptField(concept=employee_concept, field=first_name_field, order=1).save()
        DataConceptField(concept=employee_concept, field=last_name_field, order=2).save()
        DataConceptField(concept=employee_concept, field=is_manager_field, order=3).save()
        DataConceptField(concept=employee_concept, field=title_field, order=4).save()
        DataConceptField(concept=employee_concept, field=salary_field, order=5).save()

        self.concepts = [employee_concept]

        self.query = models.Employee.objects.values_list('first_name', 'last_name',
                'is_manager', 'title__name', 'title__salary')

    def test_iterator(self):
        exporter = export.Exporter(self.query, self.concepts)
        self.assertEqual(len(list(exporter)), 6)

    def test_csv(self):
        exporter = export.CSVExporter(self.query, self.concepts)
        buff = exporter.write()
        buff.seek(0)
        self.assertEqual(len(buff.read()), 246)

    def test_excel(self):
        fname = 'excel_export.xlsx'
        exporter = export.ExcelExporter(self.query, self.concepts)
        exporter.write(fname)
        self.assertTrue(os.path.exists(fname))
        # Observed slight size differences..
        l = len(open(fname).read())
        self.assertTrue(6220 <= l <= 6250)
        os.remove(fname)

    def test_sas(self):
        fname = 'sas_export.zip'
        exporter = export.SASExporter(self.query, self.concepts)
        exporter.write(fname)
        self.assertTrue(os.path.exists(fname))
        self.assertEqual(len(open(fname).read()), 1335)
        os.remove(fname)

    def test_r(self):
        fname = 'r_export.zip'
        exporter = export.RExporter(self.query, self.concepts)
        exporter.write(fname)
        self.assertTrue(os.path.exists(fname))
        self.assertEqual(len(open(fname).read()), 754)
        os.remove(fname)

    def test_json(self):
        exporter = export.JSONExporter(self.query, self.concepts)
        buff = exporter.write()
        buff.seek(0)
        self.assertEqual(len(buff.read()), 639)

    def test_html(self):
        exporter = export.HTMLExporter(self.query, self.concepts)
        template = Template("""<table>
{% for row in rows %}
    <tr>
    {% for item in row %}
        <td>{{ item.values|join:" " }}</td>
    {% endfor %}
    </tr>
{% endfor %}
</table>""")
        buff = exporter.write(template)
        buff.seek(0)
        self.assertEqual(len(buff.read()), 494)


class ResponseExportTestCase(FileExportTestCase):
    def test_csv(self):
        exporter = export.CSVExporter(self.query, self.concepts)
        response = HttpResponse()
        exporter.write(response)
        self.assertEqual(len(response.content), 246)

    def test_excel(self):
        exporter = export.ExcelExporter(self.query, self.concepts)
        response = HttpResponse()
        exporter.write(response)
        # Observed slight size differences..
        l = len(response.content)
        self.assertTrue(6220 <= l <= 6250)

    def test_sas(self):
        exporter = export.SASExporter(self.query, self.concepts)
        response = HttpResponse()
        exporter.write(response)
        self.assertEqual(len(response.content), 1335)

    def test_r(self):
        exporter = export.RExporter(self.query, self.concepts)
        response = HttpResponse()
        exporter.write(response)
        self.assertEqual(len(response.content), 754)

    def test_json(self):
        exporter = export.JSONExporter(self.query, self.concepts)
        response = HttpResponse()
        exporter.write(response)
        self.assertEqual(len(response.content), 639)

    def test_html(self):
        exporter = export.HTMLExporter(self.query, self.concepts)
        response = HttpResponse()
        template = Template("""<table>
{% for row in rows %}
    <tr>
    {% for item in row %}
        <td>{{ item.values|join:" " }}</td>
    {% endfor %}
    </tr>
{% endfor %}
</table>""")
        exporter.write(template, buff=response)
        self.assertEqual(len(response.content), 494)
