# All imports should be at the TOP of the file
from dotenv import load_dotenv
load_dotenv()

import os
from typing import Annotated
from typing_extensions import TypedDict
from serpapi.google_search import GoogleSearch
from fastapi import FastAPI          
from pydantic import BaseModel       
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from fastapi.middleware.cors import CORSMiddleware
class State(TypedDict):
    messages: Annotated[list, add_messages]
    company_name: str
    research_plan: str
    search_results: str
    final_report: str



def planner_node(state: State) -> dict:
    messages = state.get("messages", [])
    query = messages[-1].content if messages else ""
    
    llm = init_chat_model("llama-3.3-70b-versatile", model_provider="groq")
    response = llm.invoke([
        SystemMessage(content="""You are a competitive intelligence planner. 
        Given a company name, its niche, and target country, identify its top 4-5 direct competitors IN THAT COUNTRY.
        Return ONLY a numbered list of competitor names, nothing else.
        Example:
        1. Competitor A
        2. Competitor B
        3. Competitor C"""),
        HumanMessage(content=f"Company to analyze: {query}")
    ])
    return {
        "research_plan": response.content,
        "company_name": query
    }


def search_node(state: State) -> dict:
    research_plan = state.get("research_plan", "")
    company_name = state.get("company_name", "")
    
    llm = init_chat_model("llama-3.3-70b-versatile", model_provider="groq")
    extraction = llm.invoke([
        SystemMessage(content="Extract only the competitor company names from this text. Return ONLY a Python list like: ['Company1', 'Company2', 'Company3']. Nothing else."),
        HumanMessage(content=research_plan)
    ])
    
    try:
        competitor_names = eval(extraction.content.strip())
    except:
        competitor_names = ["competitor analysis"]
    
    all_results = []
    for name in competitor_names[:5]:
        try:
            search = GoogleSearch({
                "q": f"{name} pricing features reviews {company_name} competitor",
                "api_key": os.environ["SERPAPI_API_KEY"],
                "num": 3
            })
            response = search.get_dict()          # This line is fine
            for item in response.get("organic_results", []):
                all_results.append({
                    "competitor_name": name,
                    "title": item.get("title"),
                    "link": item.get("link"),
                    "snippet": item.get("snippet", "")
                })
        except Exception as e:
            all_results.append({"competitor_name": name, "error": str(e)})
    
    return {"search_results": str(all_results)}


def analyst_node(state: State) -> dict:
    llm = init_chat_model("llama-3.3-70b-versatile", model_provider="groq")
    response = llm.invoke([
        SystemMessage(content="You are a research analyst. Synthesize the search results into clear findings."),
        HumanMessage(content=f"Research plan:\n{state.get('research_plan', '')}\n\nSearch results:\n{state.get('search_results', '')}")
    ])
    return {"final_report": response.content}

def writer_node(state: State) -> dict:
    llm = init_chat_model("llama-3.3-70b-versatile", model_provider="groq")
    company_name = state.get("company_name", "the company")
    
    response = llm.invoke([
        SystemMessage(content=f"""You are a professional competitive intelligence writer.
        Write a comprehensive competitor analysis report for {company_name}.
        
        Use EXACTLY this structure with markdown:
        
        # 📊 Competitor Intelligence Report: {company_name}
        
        ## 🎯 Executive Summary
        (2-3 sentence overview)
        
        ## 🏆 Top Competitors Identified
        (bullet list of competitors)
        
        ## 📋 Detailed Competitor Analysis
        For each competitor include:
        - ✅ Strengths
        - ❌ Weaknesses  
        - 💰 Pricing (if available)
        - 🎯 Target Market
        
        ## 🔍 Market Gaps & Opportunities
        (what competitors are missing)
        
        ## 💡 Strategic Recommendations
        (actionable insights)
        
        ## 🔗 Sources
        (list all URLs referenced)"""),
        HumanMessage(content=f"Analyst findings:\n{state.get('final_report', '')}\n\nSearch results:\n{state.get('search_results', '')}")
    ])
    return {"final_report": response.content}


# ✅ What it should be
builder = StateGraph(State)
builder.add_node("planner", planner_node)
builder.add_node("search", search_node)
builder.add_node("analyst", analyst_node)
builder.add_node("writer", writer_node)
builder.add_edge(START, "planner")
builder.add_edge("planner", "search")
builder.add_edge("search", "analyst")
builder.add_edge("analyst", "writer")
builder.add_edge("writer", END)
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)



app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    company_name: str
    niche: str  
    country: str = "United States" 

class ResearchResponse(BaseModel):
    company_name: str
    research_plan: str
    final_report: str

@app.get("/")
def root():
    return {"status": "Competitor Intelligence Agent is running"}

@app.post("/analyze", response_model=ResearchResponse)
def run_research(request: AnalyzeRequest):
    config = {"configurable": {"thread_id": f"analyze-{request.company_name[:20]}"}}
    
    # Combine all input into one query
    full_query = f"Company: {request.company_name}, Niche: {request.niche}, Country: {request.country}"
    
    initial_state = {
        "messages": [HumanMessage(content=full_query)],
        "company_name": request.company_name,
        "research_plan": "",
        "search_results": "",
        "final_report": ""
    }
    
    result = graph.invoke(initial_state, config=config)
    
    return ResearchResponse(
        company_name=result.get("company_name", ""),
        research_plan=result["research_plan"],
        final_report=result["final_report"]
    )