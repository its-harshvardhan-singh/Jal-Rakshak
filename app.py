import streamlit as st
import os, sys, subprocess

# Ensure cv2 runs in headless mode without GUI/OpenGL drivers
try:
    import cv2
except (ImportError, Exception):
    # 1. Purge any corrupted cv2 reference from Python memory
    for mod in list(sys.modules.keys()):
        if mod == "cv2" or mod.startswith("cv2."):
            del sys.modules[mod]

    # 2. Overwrite GUI binaries with headless OpenCV (bypassing PEP 668 exit code 2)
    env = dict(os.environ, PIP_BREAK_SYSTEM_PACKAGES="1")
    cmd = [
        sys.executable, "-m", "pip", "install",
        "--force-reinstall", "--no-deps",
        "opencv-python-headless"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)

    # 3. Reload cv2 or display full diagnostics if pip fails
    try:
        import cv2
    except Exception as err:
        st.error(
            f"### OpenCV Setup Error\n\n"
            f"**Exit Code:** `{res.returncode}`\n\n"
            f"**STDERR:**\n```{res.stderr}```\n\n"
            f"**STDOUT:**\n```{res.stdout}```\n\n"
            f"**Import Error:** `{err}`"
        )
        st.stop()

import numpy as np
from PIL import Image
import folium
from folium.plugins import Draw, Fullscreen
from streamlit_folium import st_folium
from pipeline import app, GraphState 

st.set_page_config(page_title="Jal-Rakshak Dashboard", layout="wide")

st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #FF4B4B;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: white;
    }
    .metric-label {
        font-size: 14px;
        color: #A0A0A0;
    }
    .alert-critical {
        background-color: rgba(255, 75, 75, 0.2);
        border: 1px solid #FF4B4B;
        padding: 15px;
        border-radius: 5px;
        color: #FF4B4B;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("Jal-Rakshak: AI Maritime Surveillance")
st.markdown("Automated SAR Image Processing | YOLOv8 Segmentation | LangGraph Orchestration")
st.divider()

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Control Panel")
    
    # 1. THE NEW GEOFENCING FEATURE
    # 1. THE NEW GEOFENCING FEATURE
    # 1. THE GEOFENCING FEATURE
    st.markdown("### 🛰️ Sentinel-1 Sector Monitoring")
    st.info("Click the top-right icon to expand fullscreen for precision drawing.")
    
    # Create the map
    m = folium.Map(location=[15.0, 70.0], zoom_start=4)
    
    # Add Fullscreen Button
    Fullscreen(
        position="topright",
        title="Expand to Fullscreen",
        title_cancel="Exit Fullscreen",
        force_separate_button=True
    ).add_to(m)
    
    # Add Drawing Toolbar
    Draw(
        export=False,
        position="topleft",
        draw_options={
            "polyline": False,
            "poly": True,
            "circle": False,
            "marker": False,
            "circlemarker": False,
            "rectangle": True,
        }
    ).add_to(m)
    
    # Render the interactive map
    map_data = st_folium(m, height=260, width=300)
    
    # Capture drawn boundaries
    if map_data and map_data.get("all_drawings"):
        if len(map_data["all_drawings"]) > 0:
            geom = map_data["all_drawings"][-1]["geometry"]
            st.success("✅ Custom Geofence Activated!")
            st.markdown(f"*Monitoring {len(geom['coordinates'][0])} boundary points for new SAR passes.*")
    st.divider()
    
    # 2. Existing Manual Upload (Fallback for testing)
    st.markdown("### Manual Override (Testing)")
    uploaded_file = st.file_uploader("Upload SAR Satellite Imagery (JPG/PNG)", type=["jpg", "jpeg", "png"])
    live_api_toggle = st.toggle("🌐 Enable Live External Network Requests", value=False)

col1, col2 = st.columns([2, 1])

if uploaded_file is not None:
    temp_path = "temp_upload.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    with col1:
        st.subheader("Raw Satellite Imagery")
        image = Image.open(temp_path)
        st.image(image, use_container_width=True)
        
        if st.button("Run Pipeline", type="primary", use_container_width=True):
            with st.spinner("Initializing LangGraph Multi-Agent Workflow..."):
                
                initial_state = {
                    "image_path": temp_path,
                    "use_live_api": live_api_toggle, # NEW: Passing the toggle state to LangGraph
                    "spill_detected": False,
                    "spill_coords": [],
                    "spill_area_sq_km": 0.0,
                    "estimated_spill_volume_tons": 0.0,
                    "is_false_positive": False,
                    "suspect_vessel": {},
                    "incois_data": {},
                    "alert_status": "Pending"
                }
                
                try:
                    final_state = app.invoke(initial_state)
                    st.success("Pipeline Execution Complete!")
                    st.subheader("Processed Analysis (Instance Segmentation)")
                    
                    if final_state["spill_detected"] and len(final_state["spill_coords"]) > 0:
                        img_cv = cv2.imread(temp_path)
                        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                        
                        pts = np.array(final_state["spill_coords"], np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        
                        overlay = img_cv.copy()
                        cv2.fillPoly(overlay, [pts], (255, 0, 0))
                        cv2.addWeighted(overlay, 0.4, img_cv, 0.6, 0, img_cv)
                        cv2.polylines(img_cv, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
                        
                        st.image(img_cv, use_container_width=True, caption=f"YOLOv8 Polygon Extracted: {len(final_state['spill_coords'])} Boundary Points")
                    else:
                        st.info("No anomalies detected in this sector.")
                        
                    with col2:
                        st.subheader("Actionable Intelligence")
                        
                        if final_state["spill_detected"]:
                            st.markdown(f'<div class="alert-critical">{final_state["alert_status"]}</div>', unsafe_allow_html=True)
                            st.markdown("<br>", unsafe_allow_html=True)
                            
                            st.markdown(f'''
                            <div class="metric-card">
                                <div class="metric-label">Estimated Surface Area</div>
                                <div class="metric-value">{final_state["spill_area_sq_km"]} km²</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-label">Estimated Volume</div>
                                <div class="metric-value">{final_state["estimated_spill_volume_tons"]} Metric Tons</div>
                            </div>
                            ''', unsafe_allow_html=True)
                            
                            st.markdown("### Cross-Reference Logs")
                            st.json(final_state["incois_data"])
                            st.markdown("### Suspect Vessel ID")
                            st.json(final_state["suspect_vessel"])
                            
                        else:
                            st.success(final_state["alert_status"])
                            
                except Exception as e:
                    st.error(f"Pipeline Error: {e}")
                    st.info("Make sure best.pt and pipeline.py are in the same folder as app.py")
else:
    with col1:
        st.info("Awaiting satellite telemetry. Please upload an image from the sidebar to begin.")