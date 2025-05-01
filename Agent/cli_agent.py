# cli_agent.py
import os
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain_together import ChatTogether

from tools.vector_tool import vector_tool
from tools.graph_tool import graph_tool

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

# 🛠️ Initialize the agent with tools
agent = initialize_agent(
    tools=[vector_tool, graph_tool],
    llm=llm,
    memory=memory,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True
)

def main():
    print("🤖 Code Agent is ready. Ask me anything about your codebase.\n")
    while True:
        try:
            query = input(">> ")
            if query.lower() in ["exit", "quit"]:
                break
            response = agent.run(query)
            print(f"\n🧠 {response}\n")
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
