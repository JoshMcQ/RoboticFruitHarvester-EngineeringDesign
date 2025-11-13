# streamlit_pick_place.py
import os, sys, json, shlex, signal, threading, subprocess, queue, time
from collections import deque
import streamlit as st
from streamlit_autorefresh import st_autorefresh

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PY_EXE  = sys.executable
SCRIPT  = os.path.join(APP_DIR, "yolo_pick_place_force.py")

# ---------- Session state init (MAIN THREAD ONLY) ----------
def init_state():
    if "proc" not in st.session_state:        st.session_state.proc = None
    if "reader" not in st.session_state:      st.session_state.reader = None
    if "logq" not in st.session_state:        st.session_state.logq = queue.Queue()
    if "logbuf" not in st.session_state:      st.session_state.logbuf = deque(maxlen=2000)
    if "tail" not in st.session_state:        st.session_state.tail = 400
    if "status" not in st.session_state:      st.session_state.status = "Idle"
    if "args_snapshot" not in st.session_state: st.session_state.args_snapshot = []

init_state()

# ---------- Reader thread: NEVER touches st.* ----------
def start_reader(proc: subprocess.Popen, logq: queue.Queue):
    def _reader():
        try:
            for line in iter(proc.stdout.readline, ""):
                if not line:
                    break
                logq.put(line.rstrip("\n"))
        except Exception as e:
            logq.put(f"[reader] error: {e}")
    t = threading.Thread(target=_reader, name="log-reader", daemon=True)
    t.start()
    return t

# ---------- UI ----------
st.set_page_config(page_title="Pick & Place Controller", layout="wide")
st.markdown(
    """
    <style>
    .stButton>button { height: 42px; border-radius: 10px; }
    .small-note { color:#888; font-size:0.9em; }
    textarea { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; }
    </style>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([0.33, 0.67])

with left:
    st.header("Controls")
    targets = st.multiselect(
        "Target classes (YOLO names)",
        ["sports ball","orange","apple","lemon"],
        default=["sports ball","orange","apple","lemon"],
    )
    extra = st.text_input("Extra classes (comma-separated)", "")

    min_conf = st.slider("min confidence", 0.05, 0.95, 0.35, 0.05)
    detect_every = st.number_input("detect every N frames", 1, 60, 5, 1)

    st.subheader("Bias")
    dx = st.number_input("dx (mm)", value=0.0, step=0.1, format="%.3f")
    dy = st.number_input("dy (mm)", value=0.0, step=0.1, format="%.3f")
    dz = st.number_input("dz (mm)", value=0.0, step=0.1, format="%.3f")
    dy_pos = st.text_input("dy_pos (mm) or blank", "")
    dy_neg = st.text_input("dy_neg (mm) or blank", "")
    y_bias_per_px = st.number_input("y bias per px (mm/px)", value=0.0, step=0.001, format="%.3f")

    st.subheader("Force Sensor")
    com_port = st.text_input("COM port", "COM5")
    baud     = st.number_input("baud", 1200, 230400, 9600, 100)
    force_th = st.number_input("force threshold", 1.0, 1000.0, 100.0, 1.0)

    st.subheader("Place Bin / Travel")
    bin_x = st.number_input("bin_x", value=313.4, step=0.1, format="%.1f")
    bin_y = st.number_input("bin_y", value=-353.5, step=0.1, format="%.1f")
    bin_z = st.number_input("bin_z (place height)", value=-54.6, step=0.1, format="%.1f")
    bin_approach_z = st.number_input("bin_approach_z", value=156.1, step=0.1, format="%.1f")
    travel_z = st.number_input("travel_z", value=380.0, step=0.1, format="%.1f")

    align_to_rgb = st.checkbox("Align depth to RGB", value=True)

    st.divider()
    c1, c2, c3 = st.columns(3)
    start_clicked = c1.button("Start", type="primary", use_container_width=True, disabled=st.session_state.proc is not None)
    stop_clicked  = c2.button("Stop",  use_container_width=True, disabled=st.session_state.proc is None)
    clear_clicked = c3.button("Clear Logs", use_container_width=True)

with right:
    st.caption(f"Python: {PY_EXE} • Working dir: {APP_DIR}")
    st.header("Status")
    st.status = st.empty()
    st.status.write(st.session_state.status)

    st.subheader("Logs (tail)")
    st.session_state.tail = int(st.slider("Show last N lines", 50, 1500, st.session_state.tail, 50))
    log_box = st.empty()

    st.subheader("Command Line")
    cmd_box = st.empty()

# ---------- Build command from UI ----------
def build_cmd():
    classes = list(targets)
    if extra.strip():
        classes += [c.strip() for c in extra.split(",") if c.strip()]
    args = [
        PY_EXE, "-u", SCRIPT,
        "--conf", str(min_conf),
        "--detect-every", str(int(detect_every)),
        "--force-port", com_port,
        "--force-baud", str(int(baud)),
        "--force-threshold", str(float(force_th)),
        "--bin-x", str(bin_x),
        "--bin-y", str(bin_y),
        "--bin-z", str(bin_z),
        "--bin-approach-z", str(bin_approach_z),
        "--travel-z", str(travel_z),
    ]
    # biases
    if dx: args += ["--dx", str(dx)]
    if dy: args += ["--dy", str(dy)]
    if dz: args += ["--dz", str(dz)]
    if dy_pos.strip(): args += ["--dy-pos", dy_pos.strip()]
    if dy_neg.strip(): args += ["--dy-neg", dy_neg.strip()]
    if y_bias_per_px: args += ["--y-bias-per-px", str(y_bias_per_px)]
    if align_to_rgb: args += ["--align-to-rgb"]

    # classes list
    args += ["--classes"] + classes
    return args

cmd = build_cmd()
cmd_box.code(" ".join(shlex.quote(x) for x in cmd), language="bash")

# ---------- Start / Stop / Clear ----------
if start_clicked and st.session_state.proc is None:
    # reset logs
    st.session_state.logbuf.clear()
    with open(os.path.join(APP_DIR, "streamlit_launch.log"), "a", encoding="utf-8") as f:
        f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] CMD: {' '.join(cmd)}\n")

    # Windows: new process group so we can send CTRL_BREAK_EVENT
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

    st.session_state.proc = subprocess.Popen(
        cmd,
        cwd=APP_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
        creationflags=creationflags,
    )
    st.session_state.reader = start_reader(st.session_state.proc, st.session_state.logq)
    st.session_state.status = "Running"
    st.rerun()

if stop_clicked and st.session_state.proc is not None:
    try:
        if os.name == "nt":
            st.session_state.proc.send_signal(signal.CTRL_BREAK_EVENT)  # graceful
            time.sleep(0.6)
        st.session_state.proc.terminate()
        time.sleep(0.6)
    except Exception:
        pass
    finally:
        st.session_state.proc = None
        st.session_state.reader = None
        st.session_state.status = "Stopped"
        st.rerun()

if clear_clicked:
    st.session_state.logbuf.clear()

# ---------- Drain queue into session_state (MAIN THREAD) ----------
while not st.session_state.logq.empty():
    line = st.session_state.logq.get_nowait()
    st.session_state.logbuf.append(line)

# auto-refresh UI while the process is running
if st.session_state.proc is not None:
    st_autorefresh(interval=800, key="auto")

# ---------- Render logs & status ----------
st.status.write(st.session_state.status)
tail_lines = list(st.session_state.logbuf)[-st.session_state.tail:]
log_box.text("\n".join(tail_lines))
