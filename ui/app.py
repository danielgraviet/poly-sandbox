import streamlit as st
import requests

API_BASE = "http://localhost:8000"  # FastAPI backend URL

st.set_page_config(page_title="PolySandbox Demo", layout="wide")

# ---------------- Sidebar ---------------- #
st.sidebar.title("⚙️ PolySandbox Settings")

backend = st.sidebar.selectbox(
    "Select Backend", ["daytona", "e2b"], index=0, help="Choose which sandbox backend to use"
)
dataset = st.sidebar.selectbox("Select Dataset", ["MBPP"], index=0)
problem_index = st.sidebar.number_input(
    "Problem Index", min_value=0, max_value=119, value=0, step=1
)
st.sidebar.divider()

st.sidebar.subheader("Actions")
run_mode = st.sidebar.radio("Run Mode", ["Single Problem", "Batch Evaluation"])
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

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Problems", data["total"])
            col2.metric("Passed", data["passed"])
            col3.metric("Failed", data["failed"])
            col4.metric("Accuracy", f"{data['accuracy']*100:.1f}%")

            st.caption(
                f"Results saved to `{data['results_path']}` "
                f"(Backend: {backend.upper()})"
            )
        else:
            st.error(f"❌ Error {resp.status_code}: {resp.text}")

else:
    st.info("Select a mode and click **Run Evaluation** to start.")
