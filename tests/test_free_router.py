import pytest

import server

MODELS = {"data": [{"id": "qwen3.6-35b-a3b", "status": {"value": "loaded"}},
                   {"id": "gemma-4-12b", "status": {"value": "unloaded"}}]}


@pytest.fixture
def calls(monkeypatch):
    log = []

    def fake_request(path, body=None, timeout=1800, url=None):
        log.append((path, body))
        return MODELS if path == "/v1/models" else {}

    monkeypatch.setattr(server, "request", fake_request)
    monkeypatch.setattr(server.time, "sleep", lambda s: None)
    return log


def test_unloads_loaded_models(calls):
    unloaded = server.free_router({"router_url": "http://r:8080", "agent": {"self_model": None}})
    assert unloaded == ["qwen3.6-35b-a3b"]
    assert ("/models/unload", {"model": "qwen3.6-35b-a3b"}) in calls


def test_refuses_to_unload_the_agents_own_model(calls):
    with pytest.raises(SystemExit) as e:
        server.free_router({"router_url": "http://r:8080", "agent": {"self_model": "qwen3.6-35b-a3b"}})
    assert "stop the agent itself" in str(e.value)
    assert not any(p == "/models/unload" for p, _ in calls)


def test_no_router_configured(calls):
    assert server.free_router({"router_url": None}) == []
    assert calls == []


def test_engine_server_command(monkeypatch):
    monkeypatch.setitem(server.CFG, "models", {"q": {"alias": "qwen-x", "path": "/m/q.gguf", "ctx": 4096,
                                                     "engines": {"ex": "Qwen-exl2"}}})
    monkeypatch.setitem(server.CFG, "engines", {"ex": {"start": "/e/start.sh --port {port} --model-name {model} -c {ctx}",
                                                       "served_name": "{model}"}})
    s = server.make_server("q", ["--cache-mode", "Q8"], "ex", log="/tmp/x.log")
    assert isinstance(s, server.EngineServer)
    assert s.cmd == ["/e/start.sh", "--port", str(server.PORT), "--model-name", "Qwen-exl2", "-c", "4096",
                     "--cache-mode", "Q8"]
    assert s.alias == "Qwen-exl2"
