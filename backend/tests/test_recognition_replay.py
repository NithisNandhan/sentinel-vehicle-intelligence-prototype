"""Protected completed-session replay; no arbitrary media proxy or new events."""
import copy
import os
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

os.environ.setdefault('EDGE_API_KEY', 'test-edge-key')
os.environ.setdefault('OPERATOR_API_KEY', 'test-operator-key')
os.environ.setdefault('WATCHLIST_HMAC_KEY', 'test-hmac-key')
from app import recognition
from app.config import Settings

AUTH = {'X-Operator-Key': 'test-operator-key'}
URL = '/v1/recognition/replay?session_id=s-test'


@pytest.fixture
def client(monkeypatch):
    settings = Settings(edge_api_key='test-edge-key', operator_api_key='test-operator-key',
                        watchlist_hmac_key='test-hmac-key', dashboard_recorded_evidence=True)
    monkeypatch.setattr(recognition, 'get_settings', lambda: settings)
    import app.security
    monkeypatch.setattr(app.security, 'get_settings', lambda: settings)
    app = FastAPI(); app.include_router(recognition.router)
    with TestClient(app) as test: yield test


def page():
    return dict(schema_version='1.0', source_mode='recorded', session_id='s-test', camera_id='rec-test',
        video_duration_ms=1000, available=True, sample_count=2, offset=0, next_offset=1,
        samples=[dict(video_position_ms=0, frame='QUJD', vehicles=[dict(key='0-1', track_id='1',
            reading=None, secret='DROP', source_url='rtsp://DROP')],
            vehicles_seen=[dict(key='0-1', track_id='1', reading='DROP', masked_plate=None)],
            unique_model_accepted_unverified=0, secret='DROP')], secret='DROP')


def test_auth_opt_in_and_query_validation(client, monkeypatch):
    monkeypatch.setattr(recognition, 'request_edge', lambda *a: pytest.fail('Disabled/auth-invalid replay reached edge'))
    assert client.get(URL).status_code == 401
    for extra in ('&limit=2', '&offset=-1', '&mode=https://private'):
        assert client.get(URL+extra, headers=AUTH).status_code == 422
    recognition.get_settings().dashboard_recorded_evidence = False
    assert client.get(URL, headers=AUTH).status_code == 404


def test_historical_page_is_allowlisted_without_final_values(client, monkeypatch):
    value = page(); paths = []
    monkeypatch.setattr(recognition, 'request_edge', lambda method, path: paths.append(path) or value)
    response = client.get(URL, headers=AUTH)
    assert response.status_code == 200
    assert response.headers['cache-control'] == 'no-store, max-age=0'
    data = response.json()
    assert data['sample_count'] == 2 and data['next_offset'] == 1
    sample = data['samples'][0]
    assert sample['video_position_ms'] == 0 and sample['vehicles'][0]['reading'] is None
    assert sample['vehicles_seen'][0]['masked_plate'] is None
    assert 'DROP' not in response.text and 'rtsp' not in response.text
    assert paths == ['/internal/replay?session_id=s-test&offset=0&limit=1&mode=overlays']


@pytest.mark.parametrize('patch', [dict(source_mode='live'), dict(sample_count=301), dict(video_duration_ms=True),
    dict(offset=1), dict(next_offset=None), dict(available=False), dict(samples=[]), dict(camera_id='https://DROP')])
def test_malformed_replay_envelopes_fail_closed(client, monkeypatch, patch):
    value = page(); value.update(patch)
    monkeypatch.setattr(recognition, 'request_edge', lambda *a: value)
    assert client.get(URL, headers=AUTH).status_code == 503


def test_session_change_and_oversized_frame_are_explicit(client, monkeypatch):
    value = page(); value['session_id'] = 's-other'
    monkeypatch.setattr(recognition, 'request_edge', lambda *a: value)
    assert client.get(URL, headers=AUTH).status_code == 409
    value = page(); value['samples'][0]['frame'] = 'A'*650_001
    monkeypatch.setattr(recognition, 'request_edge', lambda *a: value)
    assert client.get(URL, headers=AUTH).status_code == 503
