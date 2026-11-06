"""Project endpoints touched by the retrieval work: paged search and index warm-up."""
import time

import config
from services import code_index


def upload(client, files):
    r = client.post("/upload-files", files=[("files", (name, body.encode())) for name, body in files.items()])
    assert r.status_code == 200, r.text
    return r.json()["project_id"]


def test_search_endpoint_pages_and_validates(client):
    pid = upload(client, {f"a{i}.py": "needle = 1\n" for i in range(5)})
    page = client.get(f"/projects/{pid}/search", params={"query": "needle", "limit": 2}).json()
    assert page["total_files"] == 5 and page["returned_files"] == 2 and page["has_more"] is True
    assert page["next_offset"] == 2
    rest = client.get(f"/projects/{pid}/search", params={"query": "needle", "limit": 10, "offset": 2}).json()
    assert rest["returned_files"] == 3 and rest["has_more"] is False
    assert client.get(f"/projects/{pid}/search", params={"query": "  "}).status_code == 400
    assert client.get(f"/projects/{pid}/search", params={"query": "x", "limit": "many"}).status_code == 422
    assert client.get("/projects/00000000-0000-4000-8000-000000000000/search",
                      params={"query": "x"}).status_code in (400, 404)


def test_uploading_warms_the_index_in_the_background(client, monkeypatch):
    monkeypatch.setattr(config, "WARM_INDEX_ON_UPLOAD", True)
    code_index.clear_cache()
    pid = upload(client, {"a.py": "def warm():\n    pass\n"})
    for _ in range(50):
        if pid in code_index._CACHE:
            break
        time.sleep(0.1)
    assert pid in code_index._CACHE


def test_warm_up_failures_never_break_an_upload(client, monkeypatch):
    monkeypatch.setattr(config, "WARM_INDEX_ON_UPLOAD", True)
    monkeypatch.setattr(code_index, "get_index", lambda pid: (_ for _ in ()).throw(RuntimeError("boom")))
    assert upload(client, {"a.py": "x = 1\n"})
