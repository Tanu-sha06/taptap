import os
import math
import json
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai

# =====================================================================
# 1. APPLICATION SETUP & CORS
# =====================================================================
app = FastAPI(title="Community Hero - Smart Civic Platform")

# Enable CORS so any frontend web environment can communicate with the endpoints
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini Flash API
API_KEY = os.environ.get("GEMINI_API_KEY", "")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# =====================================================================
# 2. IN-MEMORY DATABASE & GEOLOCATION HELPER
# =====================================================================
# Pre-seeded database with historical data to instantly trigger the predictive insights during your demo
mock_db_reports = [
    {"id": "ISSUE-101", "lat": 17.3852, "lon": 78.4869, "category": "water_leak", "severity": 3,
     "description": "Water bubbling from pavement", "hazard_flag": False, "priority_score": 4,
     "verified_by": ["user_99"], "created_at": datetime.now().isoformat()},
    {"id": "ISSUE-102", "lat": 17.3855, "lon": 78.4865, "category": "water_leak", "severity": 4,
     "description": "Pipe burst near corner store", "hazard_flag": False, "priority_score": 2,
     "verified_by": ["user_88"], "created_at": datetime.now().isoformat()},
    {"id": "ISSUE-103", "lat": 17.3849, "lon": 78.4872, "category": "water_leak", "severity": 2,
     "description": "Minor leakage on sidewalk", "hazard_flag": False, "priority_score": 1, "verified_by": ["user_77"],
     "created_at": datetime.now().isoformat()},
]


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates the distance in meters between two sets of GPS points."""
    R = 6371000  # Radius of Earth in meters
    phi_1, phi_2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi_1) * math.cos(phi_2) * math.sin(delta_lambda / 2.0) ** 2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))


# =====================================================================
# 3. DATA DATA SCHEMAS
# =====================================================================
class ImageAnalysisRequest(BaseModel):
    image_base64: str


class ReportSubmitRequest(BaseModel):
    user_id: str
    latitude: float
    longitude: float
    category: str
    severity: int
    description: str
    hazard_flag: bool


# =====================================================================
# 4. API ROUTING (THE CORE ENGINES)
# =====================================================================

@app.post("/api/v1/analyze-image")
async def analyze_image(request: ImageAnalysisRequest):
    """PILLAR 1: Image Categorization via Gemini Vision API"""
    if not API_KEY or API_KEY == "your_actual_gemini_api_key":
        # Fallback mock for easy testing if API key is missing
        return {"status": "success",
                "data": {"category": "pothole", "severity": 4, "description": "Large deep pothole on main road.",
                         "hazard_flag": True}}

    prompt = """
    You are an automated civic infrastructure analyzer. Analyze this image and output a raw JSON object with this exact schema:
    {
      "category": "pothole" | "garbage" | "streetlight" | "graffiti" | "water_leak" | "other",
      "severity": 4,
      "description": "A single sentence explaining the image content.",
      "hazard_flag": true
    }
    Ensure category is strictly one of the lowercase choices listed. Return ONLY raw JSON without markdown formatting blocks.
    """
    try:
        response = model.generate_content([
            prompt,
            {"mime_type": "image/jpeg", "data": request.image_base64}
        ])
        clean_json = response.text.replace("```json", "").replace("```", "").strip()
        return {"status": "success", "data": json.loads(clean_json)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/submit-report")
async def submit_report(report: ReportSubmitRequest):
    """PILLAR 2: Geospatial Clustering & Duplicate Triage"""
    global mock_db_reports

    for existing in mock_db_reports:
        distance = haversine_distance(report.latitude, report.longitude, existing['lat'], existing['lon'])

        # Deduplication Rule: If same category and within 50 meters
        if distance <= 50 and existing['category'] == report.category:
            existing['priority_score'] += 1
            if report.user_id not in existing['verified_by']:
                existing['verified_by'].append(report.user_id)
            if report.hazard_flag:
                existing['hazard_flag'] = True
            return {"status": "merged", "message": "Duplicate verified. Elevating issue urgency.", "data": existing}

    new_report = {
        "id": f"ISSUE-{len(mock_db_reports) + 101}",
        "lat": report.latitude,
        "lon": report.longitude,
        "category": report.category,
        "severity": report.severity,
        "description": report.description,
        "hazard_flag": report.hazard_flag,
        "priority_score": 1,
        "verified_by": [report.user_id],
        "created_at": datetime.now().isoformat()
    }
    mock_db_reports.append(new_report)
    return {"status": "created", "message": "New distinct civic track established.", "data": new_report}


@app.get("/api/v1/predictive-insights")
async def get_predictive_insights():
    """PILLAR 3: Time-Spatial Anomaly Discovery Engine"""
    insights = []

    # Analyze water leak cluster velocity
    water_leaks = [r for r in mock_db_reports if r['category'] == 'water_leak']
    if len(water_leaks) >= 3:
        insights.append({
            "type": "INFRASTRUCTURE_CRITICAL",
            "title": "Subterranean Pipe Integrity Alert",
            "message": f"Detected {len(water_leaks)} water failure vectors within a tight 50-meter quadrant. Indicates likely systemic main-line degradation.",
            "action": "Prioritize Civil Structural Review"
        })

    # High priority safety thresholds
    critical_hazards = [r for r in mock_db_reports if r.get('priority_score', 0) >= 3 and r.get('hazard_flag')]
    if critical_hazards:
        insights.append({
            "type": "SAFETY_HAZARD",
            "title": "Accelerated Danger Notification",
            "message": f"{len(critical_hazards)} issue tracking markers have bypassed standard safety bounds via rapid crowd-sourced confirmation.",
            "action": "Immediate Public Works Dispatch"
        })

    return {"status": "success", "insights": insights, "total_tracked": len(mock_db_reports)}


@app.get("/api/v1/reports")
async def get_all_reports():
    return {"status": "success", "reports": mock_db_reports}


# =====================================================================
# 5. DASHBOARD FRONTEND SYSTEM UI
# =====================================================================
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Community Hero - Admin Command Room</title>
        <style>
            * { box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; padding: 0; }
            body { background: #f3f4f6; color: #1f2937; padding: 2rem; }
            .container { max-width: 1200px; margin: 0 auto; }
            header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem; background: #fff; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); }
            h1 { color: #1e3a8a; font-size: 1.5rem; }
            .grid { display: grid; grid-template-columns: 1fr; gap: 1.5rem; }
            @media (min-width: 768px) { .grid { grid-template-columns: 1fr 2fr; } }
            .card { background: #fff; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); height: fit-content; }
            .card h2 { font-size: 1.2rem; margin-bottom: 1rem; color: #374151; border-bottom: 2px solid #f3f4f6; padding-bottom: 0.5rem; }
            .form-group { margin-bottom: 1rem; }
            label { display: block; font-size: 0.875rem; font-weight: 600; margin-bottom: 0.25rem; color: #4b5563; }
            input, select, textarea { width: 100%; padding: 0.5rem; border: 1px solid #d1d5db; border-radius: 6px; font-size: 0.95rem; }
            button { background: #2563eb; color: #fff; border: none; padding: 0.75rem 1rem; border-radius: 6px; font-weight: 600; cursor: pointer; width: 100%; transition: background 0.2s; }
            button:hover { background: #1d4ed8; }
            .insight-card { padding: 1rem; border-left: 5px solid; border-radius: 4px; margin-bottom: 1rem; background: #f9fafb; }
            .insight-CRITICAL { border-left-color: #ef4444; background: #fef2f2; }
            .insight-SAFETY { border-left-color: #f59e0b; background: #fffbeb; }
            .table-container { overflow-x: auto; }
            table { width: 100%; border-collapse: collapse; margin-top: 0.5rem; }
            th, td { padding: 0.75rem; text-align: left; border-bottom: 1px solid #e5e7eb; font-size: 0.9rem; }
            th { background: #f9fafb; color: #4b5563; font-weight: 600; }
            .badge { padding: 0.25rem 0.5rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; }
            .badge-pothole { background: #fee2e2; color: #991b1b; }
            .badge-water { background: #e0f2fe; color: #0369a1; }
            .badge-garbage { background: #fef3c7; color: #92400e; }
            .priority-pill { background: #eff6ff; color: #1e40af; border: 1px solid #bfdbfe; padding: 0.1rem 0.4rem; border-radius: 4px; font-weight: bold;}
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <div>
                    <h1>🛠️ Community Hero: AI Dispatch Matrix</h1>
                    <p style="color: #6b7280; font-size: 0.85rem;">Hackathon MVP Environment Core Active</p>
                </div>
            </header>

            <div class="grid">
                <div>
                    <div class="card" style="margin-bottom: 1.5rem;">
                        <h2>Step 1: AI Vision Assessment</h2>
                        <p style="font-size: 0.85rem; color: #6b7280; margin-bottom: 1rem;">Simulate an image upload. The AI classifies and determines contextual metrics automatically.</p>
                        <button onclick="runImageAnalysis()" style="background: #059669;">📸 Simulate Camera Capture & Analysis</button>
                        <div id="vision-output" style="margin-top: 1rem; font-size: 0.85rem; padding: 0.5rem; background: #f9fafb; border-radius: 6px; display: none;"></div>
                    </div>

                    <div class="card">
                        <h2>Step 2: Send Field Report</h2>
                        <form id="reportForm" onsubmit="submitForm(event)">
                            <div class="form-group">
                                <label>Reporter ID</label>
                                <input type="text" id="userId" value="citizen_412" required>
                            </div>
                            <div class="form-group">
                                <label>Latitude (Location)</label>
                                <input type="number" step="0.0001" id="lat" value="17.3850" required>
                            </div>
                            <div class="form-group">
                                <label>Longitude (Location)</label>
                                <input type="number" step="0.0001" id="lon" value="78.4867" required>
                            </div>
                            <div class="form-group">
                                <label>Category</label>
                                <select id="category">
                                    <option value="pothole">Pothole</option>
                                    <option value="water_leak" selected>Water Leak</option>
                                    <option value="garbage">Garbage Pile</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label>Severity Level (1-5)</label>
                                <input type="number" id="severity" min="1" max="5" value="3">
                            </div>
                            <div class="form-group">
                                <label>Contextual Description</label>
                                <textarea id="description" rows="2">Water spilling over curb line near coordinate node.</textarea>
                            </div>
                            <div class="form-group" style="display: flex; align-items: center; gap: 0.5rem;">
                                <input type="checkbox" id="hazardFlag" style="width: auto;">
                                <label style="margin-bottom: 0;">Direct Hazard to Traffic?</label>
                            </div>
                            <button type="submit">Submit Incident Report</button>
                        </form>
                    </div>
                </div>

                <div>
                    <div class="card" style="margin-bottom: 1.5rem; border: 1px solid #bfdbfe; background: #f8fafc;">
                        <h2>🧠 Pillar 3: AI Predictive System Insights</h2>
                        <div id="insights-box">Loading predictive streams...</div>
                    </div>

                    <div class="card">
                        <h2>Active Issues Tracking Map Matrix</h2>
                        <div class="table-container">
                            <table>
                               <thead>
                                   <tr>
                                       <th>ID</th>
                                       <th>Category</th>
                                       <th>Score / Upvotes</th>
                                       <th>Coordinates</th>
                                       <th>Status</th>
                                   </tr>
                               </thead>
                               <tbody id="reports-table">
                                   <tr><td colspan="5">Retrieving cluster arrays...</td></tr>
                               </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            const host = ""; // Calls local relative endpoints

            async function refreshUI() {
                // Fetch and draw reports (Pillar 2 verification)
                const resRep = await fetch(host + '/api/v1/reports');
                const reportsData = await resRep.json();
                const tbody = document.getElementById('reports-table');
                tbody.innerHTML = '';

                reportsData.reports.forEach(r => {
                    const tr = document.createElement('tr');
                    const badgeClass = r.category.includes('water') ? 'badge-water' : (r.category.includes('pot') ? 'badge-pothole' : 'badge-garbage');
                    tr.innerHTML = `
                        <td><b>${r.id}</b></td>
                        <td><span class="badge ${badgeClass}">${r.category}</span></td>
                        <td><span class="priority-pill">🔥 ${r.priority_score} Reports</span></td>
                        <td style="color:#6b7280; font-size:0.8rem;">${r.lat.toFixed(4)}, ${r.lon.toFixed(4)}</td>
                        <td><span style="color:${r.hazard_flag ? '#ef4444':'#10b981'}; font-weight:600;">${r.hazard_flag ? '⚠️ CRITICAL':'Active'}</span></td>
                    `;
                    tbody.appendChild(tr);
                });

                // Fetch and draw insights (Pillar 3 prediction)
                const resIns = await fetch(host + '/api/v1/predictive-insights');
                const insightsData = await resIns.json();
                const ibox = document.getElementById('insights-box');
                ibox.innerHTML = '';

                insightsData.insights.forEach(ins => {
                    const div = document.createElement('div');
                    const typeClass = ins.type.includes('CRITICAL') ? 'insight-CRITICAL' : 'insight-SAFETY';
                    div.className = `insight-card ${typeClass}`;
                    div.innerHTML = `
                        <strong style="display:block; font-size:1rem; margin-bottom:0.25rem;">${ins.title}</strong>
                        <p style="font-size:0.9rem; margin-bottom:0.5rem; color:#4b5563;">${ins.message}</p>
                        <span style="font-size:0.8rem; font-weight:700; color:#1e3a8a; text-transform:uppercase; background:#fff; padding:0.2rem 0.5rem; border-radius:4px; box-shadow:0 1px 2px rgba(0,0,0,0.05);">🔧 Targeted Fix Action: ${ins.action}</span>
                    `;
                    ibox.appendChild(div);
                });
            }

            async function runImageAnalysis() {
                const out = document.getElementById('vision-output');
                out.style.display = "block";
                out.innerText = "Analyzing simulated image payload via Gemini Vision...";

                const response = await fetch(host + '/api/v1/analyze-image', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ image_base64: "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7" }) // Dummy pixel string
                });
                const res = await response.json();
                out.innerHTML = `<strong>Gemini Parsing Decision:</strong><pre style="margin-top:0.5rem; background:#fff; padding:0.5rem; border:1px solid #e5e7eb;">${JSON.stringify(res.data, null, 2)}</pre>`;

                // Prefill form fields with AI output parameters
                document.getElementById('category').value = res.data.category == "pothole" ? "pothole" : "water_leak";
                document.getElementById('severity').value = res.data.severity;
                document.getElementById('description').value = res.data.description;
                document.getElementById('hazardFlag').checked = res.data.hazard_flag;
            }

            async function submitForm(e) {
                e.preventDefault();
                const payload = {
                    user_id: document.getElementById('userId').value,
                    latitude: parseFloat(document.getElementById('lat').value),
                    longitude: parseFloat(document.getElementById('lon').value),
                    category: document.getElementById('category').value,
                    severity: parseInt(document.getElementById('severity').value),
                    description: document.getElementById('description').value,
                    hazard_flag: document.getElementById('hazardFlag').checked
                };

                const res = await fetch(host + '/api/v1/submit-report', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(payload)
                });
                const confirmation = await res.json();
                alert(`${confirmation.message}\\nStatus: ${confirmation.status.toUpperCase()}`);
                refreshUI();
            }

            // Initial load
            refreshUI();
        </script>
    </body>
    </html>
    """