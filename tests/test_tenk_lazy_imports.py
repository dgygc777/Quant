"""Optional API clients should not be required for extraction-only imports."""

from __future__ import annotations

import builtins
import importlib
import sys
import unittest


class TestTenkLazyImports(unittest.TestCase):
    def test_tenk_reader_imports_without_api_clients(self):
        original_import = builtins.__import__
        saved_tenk_reader = sys.modules.pop('tenk_reader', None)
        sys.modules.pop('anthropic', None)
        sys.modules.pop('edgar', None)

        def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
            root = name.split('.')[0]
            if root in {'anthropic', 'edgar'}:
                raise ImportError(f'blocked optional dependency: {root}')
            return original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = blocked_import
        try:
            module = importlib.import_module('tenk_reader')

            self.assertTrue(module._is_valid_annual_form('10-K', include_amendments=False))
            self.assertFalse(module._is_valid_quarterly_form('8-K'))
            self.assertNotIn('anthropic', sys.modules)
            self.assertNotIn('edgar', sys.modules)
        finally:
            builtins.__import__ = original_import
            sys.modules.pop('tenk_reader', None)
            if saved_tenk_reader is not None:
                sys.modules['tenk_reader'] = saved_tenk_reader


if __name__ == '__main__':
    unittest.main()
