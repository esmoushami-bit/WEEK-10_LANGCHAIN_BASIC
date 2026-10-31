
from pathlib import Path
import pandas as pd

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Project paths
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"

DATASET_DIR.mkdir(exist_ok=True)

# Ollama model
llm = ChatOllama(model="llama3.2")

# Five prompt templates
prompts = {
    "General": """
    You are a friendly college assistant.
    Answer the question in simple English in approximately 5 lines.
    Question: {question}
    """,

    "Attendance": """
    You are an attendance assistant.
    Give helpful attendance guidance in approximately 5 lines.
    Question: {question}
    """,

    "Exams": """
    You are an exam preparation assistant.
    Give simple exam tips in approximately 5 lines.
    Question: {question}
    """,

    "Assignments": """
    You are an assignment assistant.
    Explain clearly in approximately 5 lines.
    Question: {question}
    """,

    "Career": """
    You are a career guidance assistant.
    Give useful career advice in approximately 5 lines.
    Question: {question}
    """
}


def save_chain_result(task, result):
    file_path = DATASET_DIR / "chain_result.csv"

    new_data = pd.DataFrame({
        "Task": [task],
        "Result": [result]
    })

    if file_path.exists():
        new_data.to_csv(file_path, mode="a", header=False, index=False)
    else:
        new_data.to_csv(file_path, index=False)


# Option 1: Interactive chatbot
def chatbot():
    print("\nCollege Assistant Chatbot")
    print("Type 'exit' to stop.")

    prompt = ChatPromptTemplate.from_template(prompts["General"])
    chain = prompt | llm

    while True:
        question = input("\nYou: ")

        if question.lower() == "exit":
            break

        response = chain.invoke({"question": question})
        print("\nBot:", response.content)


# Option 2: Display five prompt templates
def show_prompts():
    print("\nFive Prompt Templates")

    data = []

    for category, template in prompts.items():
        print(f"\n{category}:\n{template}")
        data.append({
            "Category": category,
            "Prompt": template
        })

    pd.DataFrame(data).to_csv(
        DATASET_DIR / "prompts.csv", index=False
    )

    print("\nPrompts saved successfully!")


# Option 3: Test 20 user queries
def test_queries():
    file_path = DATASET_DIR / "user_queries.csv"
    df = pd.read_csv(file_path)

    results = []

    for _, row in df.iterrows():
        category = row["Category"]
        question = row["Question"]

        if category not in prompts:
            category = "General"

        prompt = ChatPromptTemplate.from_template(
            prompts[category]
        )

        chain = prompt | llm
        response = chain.invoke({"question": question})

        print(f"\nQuestion: {question}")
        print("Response:", response.content)

        results.append({
            "Query_ID": row["Query_ID"],
            "Category": category,
            "Question": question,
            "Response": response.content
        })

    pd.DataFrame(results).to_csv(
        DATASET_DIR / "chatbot_responses.csv", index=False
    )

    print("\nAll queries tested successfully!")
    print("Responses saved in chatbot_responses.csv")


# Option 4: Compare basic and improved prompts
def compare_prompts():
    df = pd.read_csv(DATASET_DIR / "user_queries.csv")
    results = []

    basic_prompt = ChatPromptTemplate.from_template(
        "Answer this question in simple English in approximately 5 lines: {question}"
    )

    improved_prompt = ChatPromptTemplate.from_template(
        "You are a helpful college assistant. "
        "Give a clear, detailed and useful answer in approximately 5 lines. "
        "Question: {question}"
    )

    for _, row in df.head(5).iterrows():
        question = row["Question"]

        basic_response = (basic_prompt | llm).invoke(
            {"question": question}
        )

        improved_response = (improved_prompt | llm).invoke(
            {"question": question}
        )

        print("\nQuestion:", question)
        print("\nBasic Response:", basic_response.content)
        print("\nImproved Response:", improved_response.content)

        results.append({
            "Question": question,
            "Basic_Response": basic_response.content,
            "Improved_Response": improved_response.content
        })

    pd.DataFrame(results).to_csv(
        DATASET_DIR / "prompt_comparison.csv", index=False
    )

    print("\nPrompt comparison saved successfully!")


# Option 5: Summarization chain
def summarization_chain():
    text = """
    College education helps students develop knowledge,
    communication skills, critical thinking and confidence.
    It provides opportunities to participate in activities,
    work with others and prepare for future careers.
    Students also learn time management, problem solving
    and responsibility through their college experiences.
    """

    prompt = ChatPromptTemplate.from_template(
        "Summarize the following text in exactly 5 simple points:\n{text}"
    )

    chain = prompt | llm
    response = chain.invoke({"text": text})

    print("\nOriginal Text:\n", text)
    print("\nSummary:\n", response.content)

    save_chain_result("Summarization", response.content)


# Option 6: Question Answering chain
def qa_chain():
    context = """
    Students need communication skills, teamwork,
    problem-solving, time management, critical thinking,
    leadership and technical skills for a successful career.
    They should also develop creativity, adaptability
    and confidence.
    """

    question = input("\nEnter your question: ")

    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question using the given context.
        Give the answer in approximately 3 lines.
        Use simple English.
        If the context contains the answer, do not say
        that you do not know.

        Context: {context}

        Question: {question}

        Answer:
        """
    )

    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "question": question
    })

    print("\nQuestion:", question)
    print("\nAnswer:", response.content)

    save_chain_result("Question Answering", response.content)


# Option 7: Week 9 vs Week 10 comparison
def week_comparison():
    week9_file = (
        BASE_DIR.parent /
        "WEEK-9_CHATBOT_DEVELOPMENT" /
        "dataset" /
        "responses.csv"
    )

    week10_file = DATASET_DIR / "chatbot_responses.csv"

    if not week9_file.exists():
        print("\nWeek 9 responses.csv file not found.")
        return

    if not week10_file.exists():
        print("\nWeek 10 chatbot_responses.csv file not found.")
        return

    week9 = pd.read_csv(week9_file)
    week10 = pd.read_csv(week10_file)

    print("\nWeek 9 Responses:", len(week9))
    print("Week 10 Responses:", len(week10))

    comparison = pd.DataFrame({
        "Feature": [
            "Framework",
            "LLM Model",
            "Prompt Templates",
            "Response Logging",
            "Chain Support"
        ],
        "Week 9": [
            "Ollama",
            "Llama 3.2",
            "Basic Prompts",
            "CSV Logging",
            "Not Used"
        ],
        "Week 10": [
            "LangChain",
            "Llama 3.2",
            "Multiple Templates",
            "CSV Logging",
            "Supported"
        ]
    })

    print("\nWeek 9 vs Week 10 Comparison")
    print(comparison.to_string(index=False))

    save_chain_result(
        "Week Comparison",
        f"Week 9 responses: {len(week9)}, "
        f"Week 10 responses: {len(week10)}"
    )


# Main menu
while True:
    print("\n========== WEEK 10 LANGCHAIN CHATBOT ==========")
    print("1. Start College Assistant Chatbot")
    print("2. Show Five Prompt Templates")
    print("3. Test 20 User Queries")
    print("4. Compare Prompts")
    print("5. Summarization Chain")
    print("6. Question Answering Chain")
    print("7. Week 9 vs Week 10 Comparison")
    print("8. Exit")

    choice = input("\nEnter your choice: ")

    if choice == "1":
        chatbot()

    elif choice == "2":
        show_prompts()

    elif choice == "3":
        test_queries()

    elif choice == "4":
        compare_prompts()

    elif choice == "5":
        summarization_chain()

    elif choice == "6":
        qa_chain()

    elif choice == "7":
        week_comparison()

    elif choice == "8":
        print("\nThank you for using the chatbot!")
        break

    else:
        print("\nInvalid choice! Please try again.")
