from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    num1 : int
    num2 : int
    operation1: str
    output1 : str
    num3 : int
    num4 : int
    operation2: str
    output2 : str
    
def adder1(state : AgentState) -> AgentState:
    state['output1'] = state['num1'] + state['num2']
    return state

def subtractor1(state : AgentState) -> AgentState:
    state['output1'] = state['num1'] - state['num2']
    return state

def decide1(state : AgentState) -> AgentState:
    if state['operation1'] == "+":
        return "addition_operation"
    elif state['operation1'] == "-":
        return "subtraction_operation"
    else:
        return "invalid_operation"
    
def adder2(state : AgentState) -> AgentState:
    state['output2'] = state['num3'] + state['num4']
    return state

def subtractor2(state : AgentState) -> AgentState:
    state['output2'] = state['num3'] - state['num4']
    return state

def decide2(state : AgentState) -> AgentState:
    if state['operation2'] == "+":
        return "addition_operation2"
    elif state['operation2'] == "-":
        return "subtraction_operation2"
    else:
        return "invalid_operation"
    
graph = StateGraph(AgentState)

graph.add_node("adder1", adder1)
graph.add_node("subtractor1", subtractor1)
graph.add_node("router", lambda state:state)

graph.add_node("adder2", adder2)
graph.add_node("subtractor2", subtractor2)
graph.add_node("router2", lambda state:state)

graph.add_edge(START, "router")
graph.add_conditional_edges(
    "router",
    decide1,
    {
        "addition_operation": "adder1",
        "subtraction_operation": "subtractor1",
        "invalid_operation": END
    }
)

graph.add_edge("adder1", "router2")
graph.add_edge("subtractor1", "router2")

graph.add_conditional_edges(
    "router2",
    decide2,
    {
        "addition_operation2": "adder2",
        "subtraction_operation2": "subtractor2",
        "invalid_operation": END
    }
)

graph.add_edge("adder2",END)
graph.add_edge("subtractor2",END)

app = graph.compile()

response = app.invoke({"num1": 10, "num2": 5, "operation1": "+", "num3": 20, "num4": 7, "operation2": "-"})

print(response)