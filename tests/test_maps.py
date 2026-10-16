"""Synthetic architecture checks; no host inventories or benchmark fixtures."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from tools.maps import directories, load_nodes, render


class MapTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.template = json.loads((Path(__file__).parents[1]/'NODE.yaml').read_text())
        self.write_node('.', 'ROOT', None, [])

    def write_node(self, path, ident, parent, dependencies):
        node = copy.deepcopy(self.template)
        node.update(id=ident, parent=parent, depends_on=dependencies, tasks=[], blocked_by=[])
        node['branch'] = dict(name='main' if path == '.' else path+'/main',
                              parent=None if parent is None else 'main', status='planned')
        destination = self.root/path
        destination.mkdir(parents=True, exist_ok=True)
        (destination/'NODE.yaml').write_text(json.dumps(node))

    def test_local_evidence_is_excluded(self):
        (self.root/'.local'/'private').mkdir(parents=True)
        (self.root/'benchmarks'/'results').mkdir(parents=True)
        self.assertEqual(directories(self.root), ['.', 'benchmarks'])
        self.assertEqual(len(render(self.root)), 6)

    def test_leaf_status_propagates_to_parent(self):
        self.write_node('environment', 'ENV', '.', [])
        path = self.root/'environment'/'NODE.yaml'
        node = json.loads(path.read_text())
        before = render(self.root)['ROADMAP.md']
        node['development'] = 'paused'
        path.write_text(json.dumps(node))
        after = render(self.root)['ROADMAP.md']
        self.assertNotEqual(before, after)
        self.assertIn('paused', after)

    def test_dependency_cycle_is_rejected(self):
        self.write_node('environment', 'ENV', '.', ['hardware'])
        self.write_node('hardware', 'HW', '.', ['environment'])
        with self.assertRaisesRegex(ValueError, 'cycle'):
            load_nodes(self.root, directories(self.root))

    def test_inconsistent_parent_is_rejected(self):
        self.write_node('environment', 'ENV', 'missing', [])
        with self.assertRaisesRegex(ValueError, 'parent mismatch'):
            load_nodes(self.root, directories(self.root))

    def test_unknown_dependency_is_rejected(self):
        self.write_node('environment', 'ENV', '.', ['missing'])
        with self.assertRaisesRegex(ValueError, 'unknown dependency'):
            load_nodes(self.root, directories(self.root))

    def test_passed_verification_requires_explicit_scope(self):
        path = self.root/'NODE.yaml'
        node = json.loads(path.read_text())
        node['verification'] = 'passed'
        node.pop('verification_scope', None)
        path.write_text(json.dumps(node))
        with self.assertRaisesRegex(ValueError, 'verification scope is required'):
            load_nodes(self.root, directories(self.root))


if __name__ == '__main__':
    unittest.main()
