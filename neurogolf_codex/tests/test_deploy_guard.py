from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from neurogolf_codex.tools.deploy_guard import DeployError, deploy_model


class DeployGuardTest(unittest.TestCase):
    def test_rejects_lower_score_without_touching_existing_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            working = root / "working"
            archive = working / "archive"
            manifest = working / "solution_manifest.json"
            working.mkdir()
            (working / "task015.onnx").write_bytes(b"old-best")
            manifest.write_text(
                json.dumps(
                    {
                        "tasks": {
                            "task015": {
                                "best_score": 13.08,
                                "deployed_score": 13.08,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            candidate = root / "candidate.onnx"
            candidate.write_bytes(b"lower-score")

            with self.assertRaisesRegex(DeployError, "refusing regression"):
                deploy_model(
                    task="15",
                    model_path=candidate,
                    score=12.053,
                    source_topic=45,
                    working_dir=working,
                    manifest_path=manifest,
                    archive_dir=archive,
                )

            self.assertEqual((working / "task015.onnx").read_bytes(), b"old-best")
            self.assertFalse(archive.exists())

    def test_deploys_higher_score_and_archives_previous_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            working = root / "working"
            archive = working / "archive"
            manifest = working / "solution_manifest.json"
            working.mkdir()
            (working / "task007.onnx").write_bytes(b"old")
            manifest.write_text(
                json.dumps(
                    {
                        "tasks": {
                            "task007": {
                                "best_score": 8.771,
                                "deployed_score": 8.771,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            candidate = root / "task007_new.onnx"
            candidate.write_bytes(b"new-better")

            result = deploy_model(
                task="task007",
                model_path=candidate,
                score=11.363,
                source_topic=39,
                working_dir=working,
                manifest_path=manifest,
                archive_dir=archive,
            )

            self.assertEqual(result.task, "task007")
            self.assertEqual((working / "task007.onnx").read_bytes(), b"new-better")
            archived = list(archive.glob("task007_8.771pts_*.onnx"))
            self.assertEqual(len(archived), 1)
            self.assertEqual(archived[0].read_bytes(), b"old")
            data = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(data["tasks"]["task007"]["best_score"], 11.363)
            self.assertEqual(data["tasks"]["task007"]["source_topic"], 39)

    def test_rejects_external_data_sidecar_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "task002.onnx"
            candidate.write_bytes(b"model")
            (root / "task002.onnx.data").write_bytes(b"external")

            with self.assertRaisesRegex(DeployError, "external data sidecar"):
                deploy_model(
                    task="2",
                    model_path=candidate,
                    score=11.164,
                    source_topic=38,
                    working_dir=root / "working",
                    manifest_path=root / "working" / "solution_manifest.json",
                    archive_dir=root / "working" / "archive",
                )


if __name__ == "__main__":
    unittest.main()
