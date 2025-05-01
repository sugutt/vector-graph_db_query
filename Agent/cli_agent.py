# cli_agent.py
import os
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_together import ChatTogether
from dotenv import load_dotenv  # <-- Add dotenv
from tools.hybrid_tool import hybrid_tool  # <-- Import the new hybrid tool

load_dotenv()  # <-- Load .env

# 🔑 Environment variables must be set
llm = ChatTogether(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature=0.3,
    together_api_key=os.environ["TOGETHER_API_KEY"]
)

# 🧠 Memory setup
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 🛠️ Initialize the agent with the hybrid tool
agent_executor = initialize_agent(  # Renamed variable for clarity
    tools=[hybrid_tool],  # <-- Use only the hybrid tool
    llm=llm,
    memory=memory,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True  # Add robust error handling
)

def main():
    print("\n🤖 Codebase Agent Initialized. Ask me about the codebase!")
    print("   Type 'exit' or 'quit' to end.")

    while True:
        try:
            user_input = input("\n🧑 You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("🤖 Goodbye!")
                break

            # Use agent_executor.invoke for newer Langchain versions
            response = agent_executor.invoke({"input": user_input})
            print(f"\n🤖 Agent: {response['output']}")

        except Exception as e:
            print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()
