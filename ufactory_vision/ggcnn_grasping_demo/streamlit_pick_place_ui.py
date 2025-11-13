#!/usr/bin/env python3
"""
streamlit_pick_place_ui.py
Simple Streamlit UI to launch yolo_pick_place_force.py with configurable force threshold.

Usage:
    streamlit run streamlit_pick_place_ui.py
"""

import streamlit as st
import subprocess
import sys
import os
from pathlib import Path

# Page config
st.set_page_config(
    page_title="YOLO Pick & Place",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 YOLO Pick & Place with Force Sensing")
st.markdown("---")

# Get the directory of this script
script_dir = Path(__file__).parent
target_script = script_dir / "yolo_pick_place_force.py"

# Check if target script exists
if not target_script.exists():
    st.error(f"❌ Could not find `yolo_pick_place_force.py` in {script_dir}")
    st.stop()

# Configuration section
st.subheader("⚙️ Configuration")

col1, col2 = st.columns(2)

with col1:
    force_threshold = st.number_input(
        "Force Threshold",
        min_value=10.0,
        max_value=500.0,
        value=100.0,
        step=10.0,
        help="Stop closing gripper when force exceeds this value"
    )

with col2:
    force_port = st.text_input(
        "Force Sensor Port",
        value="COM5",
        help="Serial port for force sensor"
    )

# Fixed parameters (shown but not editable)
with st.expander("📍 Fixed Parameters (configured for your setup)"):
    st.code(f"""
Bin Position:
  X: 313.4 mm
  Y: -353.5 mm
  Z: -54.6 mm (place height)

Travel Heights:
  Approach Z: 156.1 mm
  Clear Z: 380.0 mm

Camera:
  Align to RGB: Enabled
    """, language="yaml")

st.markdown("---")

# Initialize session state for process tracking
if 'process' not in st.session_state:
    st.session_state.process = None
if 'running' not in st.session_state:
    st.session_state.running = False

# Run button
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])

with col_btn1:
    if st.button("▶️ Run", type="primary", disabled=st.session_state.running, use_container_width=True):
        # Build command
        cmd = [
            sys.executable,  # Use the same Python interpreter as Streamlit
            str(target_script),
            "--force-port", force_port,
            "--force-threshold", str(force_threshold),
            "--bin-x", "313.4",
            "--bin-y", "-353.5",
            "--bin-z", "-54.6",
            "--bin-approach-z", "156.1",
            "--travel-z", "380",
            "--align-to-rgb"
        ]
        
        st.session_state.running = True
        st.info(f"🚀 Launching pick & place process...\n\n**Command:**\n```\n{' '.join(cmd)}\n```")
        
        try:
            # Launch the process (non-blocking)
            st.session_state.process = subprocess.Popen(
                cmd,
                cwd=str(script_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            st.success("✅ Process started! Check the OpenCV window for live view.")
            st.info("""
**Controls in OpenCV window:**
- **p** - Stage pick plan from current detection
- **y** - Execute staged pick & place
- **n** - Cancel staged plan
- **s** - Save screenshot
- **q** - Quit
            """)
        except Exception as e:
            st.error(f"❌ Failed to start process: {e}")
            st.session_state.running = False
            st.session_state.process = None

with col_btn2:
    if st.button("⏹️ Stop", disabled=not st.session_state.running, use_container_width=True):
        if st.session_state.process:
            st.session_state.process.terminate()
            st.session_state.process.wait(timeout=5)
            st.session_state.process = None
        st.session_state.running = False
        st.warning("⏸️ Process stopped.")

# Status indicator
st.markdown("---")
if st.session_state.running:
    st.success("🟢 **Status:** Running")
    if st.session_state.process:
        poll = st.session_state.process.poll()
        if poll is not None:
            st.session_state.running = False
            if poll == 0:
                st.info("✅ Process completed successfully.")
            else:
                st.error(f"❌ Process exited with code {poll}")
                # Try to read stderr
                try:
                    stderr = st.session_state.process.stderr.read()
                    if stderr:
                        st.code(stderr, language="text")
                except Exception:
                    pass
            st.session_state.process = None
else:
    st.info("⚪ **Status:** Idle")

# Instructions
st.markdown("---")
st.subheader("📖 How to Use")
st.markdown("""
1. **Adjust Force Threshold** if needed (default: 100)
2. **Click Run** to start the pick & place system
3. An OpenCV window will open showing live camera feed
4. **In the OpenCV window:**
   - Press **'p'** to stage a pick from the current best detection
   - Press **'y'** to confirm and execute the staged pick & place
   - Press **'n'** to cancel a staged pick
   - Press **'q'** to quit
5. The robot will pick detected fruit and place it in the bin
6. Click **Stop** here to terminate the process if needed
""")

# Footer
st.markdown("---")
st.caption("🔧 Robotic Fruit Harvester - Engineering Design Project")
