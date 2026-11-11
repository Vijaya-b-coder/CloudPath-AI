import io
import json
import urllib.parse
from dashboard_app import CloudPathHandler, DASHBOARD_HTML, OBSERVATIONS_JSON, eval_df, predictor

class MockSocket:
    def __init__(self, data=b""):
        self._file = io.BytesIO(data)
    def makefile(self, mode, *args, **kwargs):
        return self._file

def test_request(method, path, body=b""):
    request_text = f"{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Length: {len(body)}\r\n\r\n"
    full_raw = request_text.encode('utf-8') + body
    
    mock_rfile = io.BytesIO(full_raw)
    mock_wfile = io.BytesIO()
    
    # Instantiate handler bypassing socket setup
    class DummyHandler(CloudPathHandler):
        def __init__(self):
            self.rfile = mock_rfile
            self.wfile = mock_wfile
            self.raw_requestline = self.rfile.readline()
            self.parse_request()
            if method == "GET":
                self.do_GET()
            elif method == "POST":
                self.do_POST()
    
    h = DummyHandler()
    mock_wfile.seek(0)
    response_data = mock_wfile.read()
    
    # Parse status line
    status_line = response_data.split(b"\r\n")[0].decode('utf-8')
    header_end = response_data.find(b"\r\n\r\n")
    headers_raw = response_data[:header_end].decode('utf-8')
    content = response_data[header_end+4:]
    return status_line, headers_raw, content

print("=== Running Socket-Free HTTP Handler Unit Tests ===")

# Test 1: GET /
status, headers, content = test_request("GET", "/")
assert "200" in status
assert b"CloudPath AI" in content
print("  [✓] GET / -> HTTP 200 (HTML: ", len(content), "bytes)")

# Test 2: GET /api/status
status, headers, content = test_request("GET", "/api/status")
assert "200" in status
status_json = json.loads(content.decode('utf-8'))
assert status_json["status"] == "ONLINE"
assert status_json["observations_count"] == 1448
print("  [✓] GET /api/status -> HTTP 200 (Status: ONLINE, Obs count: 1448)")

# Test 3: GET /api/satellite-observations
status, headers, content = test_request("GET", "/api/satellite-observations")
assert "200" in status
obs_json = json.loads(content.decode('utf-8'))
assert len(obs_json) == 1448
print("  [✓] GET /api/satellite-observations -> HTTP 200 (Total obs: 1448)")

# Test 4: GET /api/model-benchmarks
status, headers, content = test_request("GET", "/api/model-benchmarks")
assert "200" in status
bench_json = json.loads(content.decode('utf-8'))
assert len(bench_json) > 0
print("  [✓] GET /api/model-benchmarks -> HTTP 200 (Metrics count: ", len(bench_json), ")")

# Test 5: GET image
status, headers, content = test_request("GET", "/api/images/tracking_pair_000.png")
assert "200" in status
assert len(content) > 1000
print("  [✓] GET /api/images/tracking_pair_000.png -> HTTP 200 (Bytes: ", len(content), ")")

# Test 6: GET dynamic cloudmask render
status, headers, content = test_request("GET", "/api/cloudmask-render/0")
assert "200" in status
assert len(content) > 1000
print("  [✓] GET /api/cloudmask-render/0 -> HTTP 200 (Render bytes: ", len(content), ")")

# Test 7: POST /api/predict
req_payload = json.dumps({
    "tempC": 23.5, "maxtempC": 26, "mintempC": 21, "humidity": 94, "cloudcover": 98,
    "pressure": 1006, "precipMM": 5.2, "DewPointC": 22, "FeelsLikeC": 26, "windspeedKmph": 22,
    "WindGustKmph": 35, "winddirDegree": 240, "sunHour": 1.5, "uvIndex": 3, "visibility": 4,
    "season_enc": 3, "hour": 17, "month": 8
}).encode('utf-8')
status, headers, content = test_request("POST", "/api/predict", body=req_payload)
assert "200" in status
pred_json = json.loads(content.decode('utf-8'))
assert "rain_prediction" in pred_json
assert "rain_status" in pred_json
assert "rain_probability" in pred_json
assert "no_rain_probability" in pred_json
assert "prediction_valid" in pred_json
print("  [✓] POST /api/predict -> HTTP 200 (Rain:", pred_json["rain_label"], "Prob:", pred_json["rain_probability"], "NoRainProb:", pred_json["no_rain_probability"], "Amount:", pred_json["rainfall_amount_mm"], "mm)")

print("\nALL 7 WEB APPLICATION ENDPOINTS FULLY VERIFIED AND WORKING!")

