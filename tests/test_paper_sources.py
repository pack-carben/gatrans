import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.paper_sources import read_tables


class StrictSourceTests(unittest.TestCase):
    def test_strict_ooxml_preserves_coordinates_dates_and_shared_strings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source.xlsx'
            with zipfile.ZipFile(path, 'w') as z:
                z.writestr('xl/workbook.xml', '<workbook xmlns="http://purl.oclc.org/ooxml/spreadsheetml/main" '
                           'xmlns:r="http://purl.oclc.org/ooxml/officeDocument/relationships">'
                           '<sheets><sheet name="Figure" sheetId="1" r:id="rId1"/></sheets></workbook>')
                z.writestr('xl/_rels/workbook.xml.rels', '<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>')
                z.writestr('xl/sharedStrings.xml', '<sst xmlns="http://purl.oclc.org/ooxml/spreadsheetml/main"><si><t>TREE</t></si></sst>')
                z.writestr('xl/worksheets/sheet1.xml', '<worksheet xmlns="http://purl.oclc.org/ooxml/spreadsheetml/main"><sheetData>'
                           '<row><c r="B1" t="s"><v>0</v></c></row>'
                           '<row><c r="B2"><v>0.6853</v></c><c r="D2" t="d"><v>2024-03-02</v></c></row>'
                           '</sheetData></worksheet>')
            table = read_tables(path)['Figure']
            self.assertEqual(table.shape, (2, 4))
            self.assertIsNone(table.iloc[0, 0])
            self.assertEqual(table.iloc[0, 1], 'TREE')
            self.assertEqual(table.iloc[1, 1], .6853)
            self.assertEqual(table.iloc[1, 3], '2024-03-02')


if __name__ == '__main__':
    unittest.main()
