import streamlit as st
import requests
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from agents.runner_agent import RunnerAgent

API_BASE = "http://localhost:8000"  # FastAPI backend URL

st.set_page_config(page_title="PolySandbox Demo", layout="wide")

# ---------------- Sidebar ---------------- #
st.sidebar.title("⚙️ PolySandbox Settings")

backend = st.sidebar.selectbox(
    "Select Backend", ["daytona", "e2b", "docker"], index=0, help="Choose which sandbox backend to use"
)
dataset = st.sidebar.selectbox("Select Dataset", ["MBPP", "HumanEval", "SWE-Bench", "BCB", "Aider Polyglot"], index=0)
problem_index = st.sidebar.number_input(
    "Problem Index", min_value=0, max_value=119, value=0, step=1
)
st.sidebar.divider()

st.sidebar.subheader("Actions")
run_mode = st.sidebar.radio("Run Mode", ["Single Problem", "Batch Evaluation", "Run Agent"])
run_button = st.sidebar.button("🚀 Run Evaluation")

# ---------------- Main Page ---------------- #
st.title("🧠 PolySandbox — Unified Sandbox Orchestrator")

# Color themes per backend
BACKEND_COLORS = {"daytona": "#0078D7", "e2b": "#00C389"}

if run_button:
    # ---------------- SINGLE PROBLEM MODE ---------------- #
    if run_mode == "Single Problem":
        st.info(f"Running MBPP problem #{problem_index} on **{backend.upper()}** backend...")

        with st.spinner("Generating code and executing in sandbox..."):
            resp = requests.post(
                f"{API_BASE}/run",
                params={"index": problem_index, "backend": backend},
                timeout=120,
            )

        if resp.status_code == 200:
            data = resp.json()
            st.success(f"✅ Completed Problem #{problem_index} on {backend.upper()}")
            st.markdown(
                f"<div style='color:{BACKEND_COLORS[backend]};font-weight:bold'>"
                f"Backend: {backend.upper()}</div>",
                unsafe_allow_html=True,
            )
            st.write(f"**Runtime:** {data['runtime_ms']:.2f} ms")
            st.write(f"**Score:** {'✅ PASS' if data['score'] else '❌ FAIL'}")

            tab1, tab2, tab3 = st.tabs(["Stdout", "Stderr", "Metadata"])
            with tab1:
                st.code(data.get("stdout", "") or "(empty)", language="bash")
            with tab2:
                st.code(data.get("stderr", "") or "(empty)", language="bash")
            with tab3:
                st.json(data.get("metadata", {}))
        else:
            st.error(f"❌ Error {resp.status_code}: {resp.text}")

    # ---------------- BATCH MODE ---------------- #
    elif run_mode == "Batch Evaluation":
        n = st.sidebar.slider("Number of Problems", min_value=1, max_value=120, value=10)
        concurrency = st.sidebar.slider("Concurrency", min_value=1, max_value=5, value=5)

        st.info(
            f"Running MBPP batch of {n} problems "
            f"with concurrency={concurrency} on **{backend.upper()}** backend..."
        )

        with st.spinner("Running evaluation, please wait..."):
            resp = requests.post(
                f"{API_BASE}/run_batch",
                params={"n": n, "concurrency": concurrency, "backend": backend},
                timeout=300,
            )

        if resp.status_code == 200:
            data = resp.json()
            st.success(f"✅ Batch completed on {backend.upper()} backend!")

            # Show metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Total Problems", data["total"])
            col2.metric("Passed", data["passed"])
            col3.metric("Failed", data["failed"])
            col4.metric("Accuracy", f"{data['accuracy']*100:.1f}%")
            col5.metric("Total Runtime", f"{data.get('total_runtime_sec', 0):.2f}s")

            # Show average time under caption
            st.caption(
                f"Results saved to `{data['results_path']}` "
                f"(Backend: {backend.upper()}) — "
                f"Avg per problem: {data.get('avg_runtime_sec', 0):.2f}s"
            )
        else:
            st.error(f"❌ Error {resp.status_code}: {resp.text}")

    elif run_mode == "Run Agent":
        st.info("🤖 The RunnerAgent will automatically decide the backend based on problem complexity...")

        with st.spinner("Agent deciding and running..."):
            try:
                resp = requests.post(f"{API_BASE}/run_agent", params={"index": problem_index}, timeout=180)
            except requests.exceptions.RequestException as e:
                st.error(f"Network error: {e}")
                st.stop()

        if resp.status_code == 200:
            data = resp.json()
            backend = data["backend"]
            st.success(f"✅ Agent completed Problem #{problem_index} using {backend.upper()} backend")

            st.markdown(
                f"<div style='color:{BACKEND_COLORS.get(backend, 'gray')};font-weight:bold'>"
                f"Backend Chosen: {backend.upper()}</div>",
                unsafe_allow_html=True,
            )

            st.write(f"**Runtime:** {data['runtime_ms']:.2f} ms")
            st.write(f"**Score:** {'✅ PASS' if data['score'] else '❌ FAIL'}")

            tab1, tab2, tab3 = st.tabs(["Stdout", "Stderr", "Metadata"])
            with tab1:
                st.code(data.get("stdout", "") or "(empty)", language="bash")
            with tab2:
                st.code(data.get("stderr", "") or "(empty)", language="bash")
            with tab3:
                st.json(data.get("metadata", {}))

        else:
            st.error(f"❌ Error {resp.status_code}: {resp.text}")


else:
    st.info("Select a mode and click **Run Evaluation** to start.")
