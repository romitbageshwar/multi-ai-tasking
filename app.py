import streamlit as st
from huggingface_hub import InferenceClient

# Use Hugging Face hosted model (free tier)
HF_MODEL = "mistralai/Mistral-7B-Instruct"
client = InferenceClient(model=HF_MODEL)

# ----- Employee AI -----
class EmployeeAI:
    def __init__(self, role):
        self.role = role

    def work(self, task):
        response = client.text_generation(
            f"You are an AI {self.role}. Complete this task:\n{task}",
            max_new_tokens=300,
        )
        return response

# ----- Project Manager AI -----
class ProjectManagerAI:
    def __init__(self, employees):
        self.employees = employees

    def breakdown_project(self, project_description):
        response = client.text_generation(
            f"You are a project manager. Break down this project into 3 clear tasks:\n{project_description}",
            max_new_tokens=150,
        )
        return [t.strip("-• ") for t in response.split("\n") if t.strip()]

    def assign_and_collect(self, tasks):
        results = []
        for task, employee in zip(tasks, self.employees):
            result = employee.work(task)
            results.append((employee.role, task, result))
        return results

    def combine_results(self, results):
        combined_text = "\n\n".join(
            [f"### {role} ({task}):\n{res}" for role, task, res in results]
        )
        final = client.text_generation(
            f"Combine the following work into a polished final output:\n{combined_text}",
            max_new_tokens=400,
        )
        return final

# ----- Streamlit UI -----
st.set_page_config(page_title="AI Project Sync Space", layout="wide")

st.title("🤖 AI Project Sync Space")
st.write("An AI Project Manager assigns tasks to AI Employees, collects results, and produces a final deliverable.")

project_description = st.text_area("Enter your project description:", height=100)

if st.button("Run Project"):
    if project_description.strip():
        employees = [
            EmployeeAI("Researcher"),
            EmployeeAI("Writer"),
            EmployeeAI("Editor"),
        ]
        manager = ProjectManagerAI(employees)

        st.subheader("📋 Task Breakdown")
        tasks = manager.breakdown_project(project_description)
        for i, task in enumerate(tasks):
            st.write(f"**Task {i+1}:** {task}")

        st.subheader("👷 Employee Outputs")
        results = manager.assign_and_collect(tasks)
        for role, task, output in results:
            with st.expander(f"{role} - {task}"):
                st.write(output)

        st.subheader("✅ Final Output")
        final_output = manager.combine_results(results)
        st.success(final_output)
    else:
        st.warning("Please enter a project description.")
