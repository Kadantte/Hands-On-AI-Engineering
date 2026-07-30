from dotenv import load_dotenv
import streamlit as st

from crew import run_instagram_post_crew

load_dotenv()


def extract_final_section(raw_text: str, markers):
    if not raw_text:
        return raw_text

    lowered = raw_text.lower()
    for marker in markers:
        idx = lowered.find(marker)
        if idx != -1:
            candidate = raw_text[idx:].strip()
            if candidate:
                return candidate

    return raw_text.strip()


def get_task_raw_output(task_output):
    if task_output is None:
        return ""
    if hasattr(task_output, "raw"):
        return task_output.raw or ""
    return str(task_output)


st.set_page_config(page_title="Instagram Post Crew", page_icon="📸", layout="wide")

st.title("Instagram Post Crew")
st.caption("A CrewAI multi-agent pipeline that turns a topic into a full Instagram content package.")

topic = st.text_input("Enter your topic or product idea")

run_clicked = st.button("Run")

if run_clicked:
    if not topic.strip():
        st.warning("Please enter a topic or product idea before running the crew.")
    else:
        with st.spinner("Agents working..."):
            try:
                result, crew = run_instagram_post_crew(topic)
            except Exception as error:
                st.error(f"The crew run failed: {error}")
                result, crew = None, None

        if crew is not None:
            task_outputs = list(crew.tasks_output) if hasattr(crew, "tasks_output") else []

            if len(task_outputs) < 4:
                task_outputs = [task.output for task in crew.tasks]

            trend_output = get_task_raw_output(task_outputs[0]) if len(task_outputs) > 0 else ""
            caption_output = get_task_raw_output(task_outputs[1]) if len(task_outputs) > 1 else ""
            visual_output = get_task_raw_output(task_outputs[2]) if len(task_outputs) > 2 else ""
            posting_output = get_task_raw_output(task_outputs[3]) if len(task_outputs) > 3 else ""

            st.header("1. Trend Intelligence Brief")
            with st.expander("Show agent reasoning"):
                st.markdown(trend_output if trend_output else "No output produced.")
            st.success(trend_output if trend_output else "No output produced.")

            st.header("2. Caption (with hashtags)")
            with st.expander("Show agent reasoning"):
                st.markdown(caption_output if caption_output else "No output produced.")
            final_caption = extract_final_section(
                caption_output,
                ["final caption", "selected caption", "final selected caption"]
            )
            st.success(final_caption if final_caption else "No output produced.")

            st.header("3. Image Generation Prompt")
            with st.expander("Show agent reasoning"):
                st.markdown(visual_output if visual_output else "No output produced.")
            final_prompt = extract_final_section(
                visual_output,
                ["final image generation prompt", "final prompt", "image generation prompt:"]
            )
            st.success(final_prompt if final_prompt else "No output produced.")

            st.header("4. Posting Strategy")
            with st.expander("Show agent reasoning"):
                st.markdown(posting_output if posting_output else "No output produced.")
            final_posting = extract_final_section(
                posting_output,
                ["final recommendation", "recommendation:"]
            )
            st.success(final_posting if final_posting else "No output produced.")
