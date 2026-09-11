import unittest

from scripts.audit_dataset_splits import audit


def row(episodes, maps, keys, starts):
    return {'episodes': episodes, 'maps': maps,
            'unique_target_description_keys': len(keys), 'unique_starts': len(starts),
            '_keys': set(keys), '_starts': set(starts)}


class DatasetSplitAuditTest(unittest.TestCase):
    def test_accepts_seen_maps_but_rejects_unseen_overlap(self):
        rows = {
            'train_seen': row(1, ['seen'], [('seen', 1, 1)], [('seen', 0)]),
            'val_seen': row(1, ['seen'], [('seen', 2, 1)], [('seen', 1)]),
            'val_unseen': row(1, ['unseen-a'], [('unseen-a', 1, 1)], [('unseen-a', 0)]),
            'test_unseen': row(1, ['unseen-b'], [('unseen-b', 1, 1)], [('unseen-b', 0)]),
        }
        checks, _ = audit(rows)
        self.assertTrue(all(checks.values()))
        rows['test_unseen']['maps'] = ['unseen-a']
        checks, _ = audit(rows)
        self.assertFalse(checks['test_unseen_maps_disjoint_from_train_and_validation'])


if __name__ == '__main__':
    unittest.main()
