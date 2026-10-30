import unittest
from bootstrap import AXES, parse_catalog, split_for_address, encode_address, decode_address

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
    def test_crystal_codec_bounds(self):
        low = ''.join(axis[2][0] for axis in AXES)
        high = ''.join(axis[2][-1] for axis in AXES)
        self.assertEqual(0, encode_address(low))
        self.assertEqual(17_279_999, encode_address(high))
        self.assertEqual(low, decode_address(0))
        self.assertEqual(high, decode_address(17_279_999))

    def test_crystal_codec_roundtrip(self):
        for n in [0, 1, 48, 1234, 6687051, 12089822, 17279999]:
            self.assertEqual(n, encode_address(decode_address(n)))

    def test_invalid_codec_inputs(self):
        with self.assertRaises(ValueError):
            decode_address(17_280_000)
        with self.assertRaises(ValueError):
            encode_address("𐑦")

    def test_split_stable(self):
        addr='𐑦𐑸𐑾𐑹𐑐𐑧𐑲𐑠⊙𐑫𐑳𐑭'
        self.assertEqual(split_for_address(addr),split_for_address(addr))

if __name__=='__main__':unittest.main()
