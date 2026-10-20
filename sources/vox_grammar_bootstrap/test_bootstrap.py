import unittest
from bootstrap import AXES, parse_catalog, split_for_address

class Tests(unittest.TestCase):
    def test_all_type_glyphs_unique(self):
        self.assertEqual(49, len(set(''.join(a[2] for a in AXES))))
    def test_valid(self):
        text='x │ ⟨𐑦𐑸𐑾𐑹𐑐𐑧𐑲𐑠⊙𐑫𐑳𐑭⟩ │ 𐑦'
        rows,errors=parse_catalog(text)
        self.assertEqual((1,0),(len(rows),len(errors)))
        self.assertEqual([4,5,4,5,3,3,3,3,2,4,3,3], rows[0]['ordinals'])
    def test_reject_cross_axis_glyph(self):
        rows,errors=parse_catalog('x │ ⟨𐑦𐑦𐑾𐑹𐑐𐑧𐑲𐑠⊙𐑫𐑳𐑭⟩ │ 𐑦')
        self.assertEqual(0,len(rows))
        self.assertTrue(errors)
    def test_split_stable(self):
        addr='𐑦𐑸𐑾𐑹𐑐𐑧𐑲𐑠⊙𐑫𐑳𐑭'
        self.assertEqual(split_for_address(addr),split_for_address(addr))

if __name__=='__main__':unittest.main()
