import streamlit as st
from multi_agent_hierarchy import graph  
from langchain_core.messages import HumanMessage

st.title("Multi-Agent AI Report Generator")

st.write(
    "Ask a question or specify a task, and the multi-agent AI system will generate a detailed report."
)

# Text input for user question/task
user_question = st.text_area("Enter your question or task:", height=100)

if st.button("Generate Report"):
    if not user_question.strip():
        st.warning("Please enter a question or task before generating a report.")
    else:
        with st.spinner("Generating report... This may take some time."):
            # Run the multi-agent graph with user input
            result = graph.invoke({
                "messages": [HumanMessage(content=user_question)],
                "current_task": user_question,
            })

            final_report = result.get("final_report")
            if final_report:
                st.subheader("Final Report")
                st.markdown(final_report)
            else:
                st.error("Failed to generate report. Please try again.")
