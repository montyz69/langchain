from typing import TypedDict
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    name : str
    age : int
    skills : list[str]
    output : str
    
def first_Node(state : AgentState) -> AgentState:
    state["output"] = f"{state['name']}, welcome to the system! "
    return state

def second_Node(state : AgentState) -> AgentState:
    state["output"] = state["output"] + f"You are {state['age']} years old! "
    return state

def third_Node(state : AgentState) -> AgentState:
    state["output"] = state["output"] + f"You have the skills in: {state['skills']}. "
    return state


graph = StateGraph(AgentState)
graph.add_node("first_Node", first_Node)
graph.add_node("second_Node", second_Node)
graph.add_node("third_Node", third_Node)

graph.set_entry_point("first_Node")
graph.add_edge("first_Node", "second_Node")
graph.add_edge("second_Node", "third_Node")
graph.set_finish_point("third_Node")

app = graph.compile()

response = app.invoke({"name": "John", "age": 30, "skills": ["coding", "reading"]})

# print(response)

print(response["output"])