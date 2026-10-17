"""The UI's index page follows the build on disk (`web_server/backend/app.py`):
a frontend rebuild while the app runs must not leave workers serving an
index whose scripts the rebuild deleted."""


def test_the_index_page_is_reread_after_a_rebuild(monkeypatch, tmp_path):
    # Imported here, not at module level: importing the web app while pytest
    # COLLECTS leaked state into tests that run before this one in the same
    # process (mcp_server/test_server.py, test_inflight_dedup.py, 2026-10-17).
    import web_server.backend.app as app_module
    (tmp_path / "index.html").write_text('<script src="/static/js/main.old.js"></script>')
    monkeypatch.setattr(app_module, "_FRONTEND_BUILD", str(tmp_path))
    client = app_module.app.test_client()
    first = client.get("/")
    assert b"main.old.js" in first.data and first.headers["Cache-Control"] == "no-cache"
    (tmp_path / "index.html").write_text('<script src="/static/js/main.new.js"></script>')
    assert b"main.new.js" in client.get("/").data
    assert b"main.new.js" in client.get("/target/abc").data      # the router's URLs too
