import unittest

from multiagent.continue_on_recovery import check_coverage, epoch_sr


def complete_record():
    lines = []
    for epoch in (0, 1):
        lines.append(
            f"EPOCH_TIMING epoch={epoch} train_seconds=100.00 "
            "train_samples=21877 train_batches=1368"
        )
        for split, count in (("val_seen", 2470), ("val_unseen", 2697)):
            lines.append(
                f"VALIDATION_TIMING epoch={epoch} split={split} "
                f"seconds=50.00 samples={count}"
            )
        lines.append(f"epoch {epoch}")
        lines.append("val_seen , ne: 50.00, sr: 26.60, spl: 20.00")
        lines.append("val_unseen , ne: 60.00, sr: 13.76, spl: 10.00")
    return "\n".join(lines)


class ContinuationGateTest(unittest.TestCase):
    def test_complete_epochs_and_validation(self):
        record = complete_record()
        check_coverage(record)
        self.assertEqual(epoch_sr(record, 1), {"val_seen": 26.60, "val_unseen": 13.76})

    def test_repeated_epoch_is_rejected(self):
        record = complete_record()
        duplicate = "EPOCH_TIMING epoch=1 train_seconds=100.00 train_samples=21877 train_batches=1368"
        with self.assertRaises(ValueError):
            check_coverage(record + "\n" + duplicate)

    def test_missing_sample_is_rejected(self):
        record = complete_record().replace("train_samples=21877", "train_samples=21876", 1)
        with self.assertRaises(ValueError):
            check_coverage(record)


if __name__ == "__main__":
    unittest.main()
