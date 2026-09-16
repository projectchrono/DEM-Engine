"""Check PyPI README preparation without building the CUDA extension."""

import importlib.util
from email.parser import BytesParser
from pathlib import Path
import tempfile
import unittest


REPOSITORY = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("prepare_pypi_readme", REPOSITORY / "docs/prepare_pypi_readme.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PyPIReadmeTests(unittest.TestCase):
    """Cover link portability and preservation of the release description."""

    def test_files_directories_anchors_and_remote_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs").mkdir()
            (root / "docs/guide.rst").touch()
            readme = root / "README.md"
            unchanged = '# DEME 3\nmesh–mesh\n[Section](#section)\n[Site](https://example.com/)\n<img src="https://example.com/demo.gif">\n'
            readme.write_text(unchanged + '[Guide](docs/guide.rst#usage)\n[Demos](docs)\n', encoding="utf-8")
            module.prepare_readme(readme, "https://github.com/example/DEME/", "release/3")
            prepared = readme.read_text(encoding="utf-8")
            self.assertTrue(prepared.startswith(unchanged))
            self.assertIn('https://github.com/example/DEME/blob/release%2F3/docs/guide.rst#usage', prepared)
            self.assertIn('https://github.com/example/DEME/tree/release%2F3/docs', prepared)
            module.prepare_readme(readme, "https://github.com/example/DEME", "other")
            self.assertEqual(readme.read_text(encoding="utf-8"), prepared)
            # Wheel METADATA has UTF-8 text without a MIME charset; decode bytes explicitly.
            metadata = BytesParser().parsebytes(
                ('Metadata-Version: 2.1\nDescription-Content-Type: text/markdown\n\n' + prepared).encode('utf-8')
            )
            self.assertEqual(metadata.get_payload(decode=True).decode('utf-8'), prepared)

    def test_missing_link_fails_without_rewriting(self):
        with tempfile.TemporaryDirectory() as directory:
            readme = Path(directory) / "README.md"
            source = '[Missing](missing.rst)\n'
            readme.write_text(source)
            with self.assertRaisesRegex(ValueError, "does not exist"):
                module.prepare_readme(readme, "https://github.com/example/DEME", "abc123")
            self.assertEqual(readme.read_text(), source)


if __name__ == "__main__":
    unittest.main()
