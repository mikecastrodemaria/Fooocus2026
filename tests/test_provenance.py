"""custom-23 — la declaration IA ecrite dans PNG / JPEG / WEBP se relit, sans prompt.

Run:  py -3.10 -m unittest tests.test_provenance -v
"""
import io
import os
import shutil
import tempfile
import unittest
from unittest import mock

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from modules import provenance as P


def fooocus_exif(parameters):
    """Meme EXIF que meta_parser.get_exif, sans l'importer (il charge args_manager)."""
    exif = Image.Exif()
    exif[0x9286] = parameters  # UserComment
    exif[0x0131] = 'Fooocus v2026'  # Software
    exif[0x927C] = 'fooocus'  # MakerNote
    return exif


def picture():
    img = Image.new('RGB', (32, 24), 'white')
    img.putpixel((3, 4), (255, 0, 0))
    return img


class TestWriteAndRead(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix='fprov_')

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_png_keeps_the_parameters_and_declares_ai(self):
        info = PngInfo()
        info.add_text('parameters', '{"Prompt": "a cat"}')
        path = os.path.join(self.dir, 'a.png')
        picture().save(path, pnginfo=P.add_to_pnginfo(info, generator='Fooocus2026 test'))
        d = P.describe(path, check_watermark=False)
        self.assertIn('declared', d['ai_generated'])
        self.assertEqual(d['generator'], 'Fooocus2026 test')
        with Image.open(path) as im:
            self.assertEqual(im.info['parameters'], '{"Prompt": "a cat"}')
            self.assertNotIn('a cat', im.info[P.PNG_XMP_KEY], 'aucun prompt dans la declaration')

    def test_png_without_metadata_still_declares_ai(self):
        path = os.path.join(self.dir, 'b.png')
        picture().save(path, pnginfo=P.add_to_pnginfo(None))
        self.assertIn('declared', P.describe(path, check_watermark=False)['ai_generated'])

    def test_jpeg_segment_is_inserted_after_exif_and_the_image_still_decodes(self):
        path = os.path.join(self.dir, 'c.jpg')
        picture().save(path, quality=95, exif=fooocus_exif('{"Prompt": "x"}'))
        self.assertTrue(P.inject_jpeg_xmp(path, generator='Fooocus2026 test'))
        self.assertTrue(P.inject_jpeg_xmp(path), 'idempotent')
        with open(path, 'rb') as f:
            self.assertEqual(f.read().count(P.JPEG_XMP_HEADER), 1)
        with Image.open(path) as im:
            im.load()
            self.assertEqual(im.size, (32, 24))
            self.assertEqual(im.getexif().get(0x9286), '{"Prompt": "x"}', 'EXIF de Fooocus intact')
        d = P.describe(path, check_watermark=False)
        self.assertIn('declared', d['ai_generated'])
        self.assertEqual(d['generator'], 'Fooocus2026 test')

    def test_webp_declares_ai(self):
        path = os.path.join(self.dir, 'd.webp')
        picture().save(path, quality=95, **P.webp_save_kwargs('Fooocus2026 test'))
        self.assertIn('declared', P.describe(path, check_watermark=False)['ai_generated'])

    def test_an_uploaded_pil_image_is_read_too(self):
        buf = io.BytesIO()
        picture().save(buf, format='PNG', pnginfo=P.add_to_pnginfo(None))
        buf.seek(0)
        self.assertIn('declared', P.describe(Image.open(buf), check_watermark=False)['ai_generated'])

    def test_an_unmarked_image_is_never_called_authentic(self):
        path = os.path.join(self.dir, 'e.png')
        picture().save(path)
        d = P.describe(path, check_watermark=False)
        self.assertEqual(d['ai_generated'], 'no machine-readable declaration found')
        self.assertEqual(d['note'], P.NOT_A_PROOF)

    def test_a_non_jpeg_file_is_left_alone(self):
        path = os.path.join(self.dir, 'f.png')
        picture().save(path)
        with open(path, 'rb') as f:
            before = f.read()
        self.assertFalse(P.inject_jpeg_xmp(path))
        with open(path, 'rb') as f:
            self.assertEqual(f.read(), before)


class TestWatermarkGate(unittest.TestCase):
    def test_watermark_off_returns_the_same_image(self):
        img = picture()
        with mock.patch.object(P, '_setting', side_effect=lambda k, d: False if k == 'watermark' else d):
            self.assertIs(P.maybe_watermark(img), img)

    def test_watermark_on_without_trustmark_never_breaks_the_save(self):
        img = picture()
        with mock.patch.object(P, '_setting', side_effect=lambda k, d: True if k == 'watermark' else d), \
                mock.patch.object(P, 'trustmark_available', return_value=False):
            self.assertIs(P.maybe_watermark(img), img)

    def test_watermark_id_is_ascii_and_short(self):
        with mock.patch.object(P, '_setting', side_effect=lambda k, d: 'Fooocus2026-éé' if k == 'watermark_id' else d):
            self.assertEqual(P.watermark_id(), 'Fooocus20')


if __name__ == '__main__':
    unittest.main()
