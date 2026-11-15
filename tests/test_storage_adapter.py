from __future__ import annotations

import tempfile
import pytest
from pathlib import Path

from storage.local import LocalStorageAdapter
from storage.s3 import S3StorageAdapter
from storage.factory import get_storage_adapter


def test_local_storage_adapter_crud():
    with tempfile.TemporaryDirectory() as tmp_dir:
        adapter = LocalStorageAdapter(base_path=tmp_dir)

        # 1. Save bytes & text
        uri_txt = adapter.save("docs/readme.txt", "Hello World AutoML")
        uri_bin = adapter.save("models/model.pkl", b"\x80\x04\x95\x05")

        assert adapter.exists("docs/readme.txt")
        assert adapter.exists("models/model.pkl")
        assert not adapter.exists("models/nonexistent.pkl")

        # 2. Load
        assert adapter.load_text("docs/readme.txt") == "Hello World AutoML"
        assert adapter.load("models/model.pkl") == b"\x80\x04\x95\x05"

        # 3. URL
        url = adapter.get_url("docs/readme.txt")
        assert url.startswith("file:///")

        # 4. Delete
        assert adapter.delete("docs/readme.txt") is True
        assert adapter.exists("docs/readme.txt") is False
        assert adapter.delete("docs/readme.txt") is False


def test_s3_storage_adapter_mock_mode():
    adapter = S3StorageAdapter(
        bucket="my-automl-bucket",
        endpoint_url="",
        region="us-west-2",
        prefix="artifacts",
    )

    # 1. Save
    uri = adapter.save("exp_123/pipeline.pkl", b"SERIALIZED_PIPELINE_DATA")
    assert uri == "s3://my-automl-bucket/artifacts/exp_123/pipeline.pkl"
    assert adapter.exists("exp_123/pipeline.pkl") is True
    assert adapter.exists("s3://my-automl-bucket/artifacts/exp_123/pipeline.pkl") is True

    # 2. Load
    loaded = adapter.load("exp_123/pipeline.pkl")
    assert loaded == b"SERIALIZED_PIPELINE_DATA"

    # 3. URL
    url = adapter.get_url("exp_123/pipeline.pkl")
    assert "my-automl-bucket" in url

    # 4. Delete
    deleted = adapter.delete("exp_123/pipeline.pkl")
    assert deleted is True
    assert adapter.exists("exp_123/pipeline.pkl") is False


def test_storage_factory():
    local_adapter = get_storage_adapter(backend="local", force_new=True)
    assert isinstance(local_adapter, LocalStorageAdapter)

    s3_adapter = get_storage_adapter(backend="s3", force_new=True)
    assert isinstance(s3_adapter, S3StorageAdapter)
