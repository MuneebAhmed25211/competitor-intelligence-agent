from dotenv import load_dotenv
load_dotenv()

import os
import random
from typing import Annotated
from typing_extensions import TypedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from serpapi.google_search import GoogleSearch
from fastapi import FastAPI, HTTPException
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
    niche: str
    country: str
    competitor_level: str
    research_plan: str
    search_results: str
    final_report: str

# ── GROQ KEY ROTATION ──
GROQ_KEYS = [
    os.environ.get("GROQ_API_KEY_1", ""),
    os.environ.get("GROQ_API_KEY_2", ""),
    os.environ.get("GROQ_API_KEY_3", ""),
]
GROQ_KEYS = [k for k in GROQ_KEYS if k]

# ── SERPAPI KEY ROTATION ──
SERP_KEYS = [
    os.environ.get("SERPAPI_API_KEY_1", ""),
    os.environ.get("SERPAPI_API_KEY_2", ""),
]
SERP_KEYS = [k for k in SERP_KEYS if k]

def get_llm():
    api_key = random.choice(GROQ_KEYS)
    return init_chat_model(
        "llama-3.3-70b-versatile",
        model_provider="groq",
        api_key=api_key
    )

def get_serp_key():
    return random.choice(SERP_KEYS)

def planner_node(state: State) -> dict:
    messages = state.get("messages", [])
    query = messages[-1].content if messages else ""
    competitor_level = state.get("competitor_level", "All sizes")

    level_instructions = {
        "All sizes": "Identify the top 5 direct competitors regardless of company size.",
        "Enterprise (Market leaders)": "Identify the top 5 largest, most established competitors — Fortune 500 level or market leaders with massive brand recognition.",
        "Mid-market (Established brands)": "Identify 5 mid-sized established competitors — companies with solid market presence but not industry giants. Revenue roughly $10M-$500M range.",
        "Small business (Startups & emerging)": "Identify 5 small or emerging competitors — startups, bootstrapped companies, or newer brands at an early growth stage with limited market share.",
    }

    instruction = level_instructions.get(competitor_level, level_instructions["All sizes"])

    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=f"""You are a competitive intelligence planner. 
        Given a company name, its niche, and target country, identify its top 5 direct competitors IN THAT COUNTRY.
        
        Competitor level filter: {competitor_level}
        Instruction: {instruction}
        
        Return ONLY a numbered list of competitor names, nothing else.
        Example:
        1. Competitor A
        2. Competitor B
        3. Competitor C"""),
        HumanMessage(content=f"Company to analyze: {query}")
    ])
    return {
        "research_plan": response.content,
        "company_name": state.get("company_name", ""),
        "niche": state.get("niche", ""),
        "country": state.get("country", "United States"),
        "competitor_level": competitor_level
    }

def search_single_competitor(args):
    name, company_name, niche, country = args
    # Try each SERP key until one works
    for api_key in SERP_KEYS:
        try:
            search = GoogleSearch({
                "q": f"{name} {niche} pricing features reviews {country}",
                "api_key": api_key,
                "num": 3
            })
            response = search.get_dict()
            
            # Check if we hit the rate limit on this key
            if "error" in response and "limit" in str(response.get("error", "")).lower():
                continue  # Try next key
                
            results = []
            for item in response.get("organic_results", []):
                results.append({
                    "competitor_name": name,
                    "title": item.get("title"),
                    "link": item.get("link"),
                    "snippet": item.get("snippet", "")
                })
            return results
        except Exception:
            continue  # Try next key
    
    # All keys failed
    return [{"competitor_name": name, "error": "search_limit_reached"}]

def search_node(state: State) -> dict:
    research_plan = state.get("research_plan", "")
    company_name = state.get("company_name", "")
    niche = state.get("niche", "")
    country = state.get("country", "United States")

    llm = get_llm()
    extraction = llm.invoke([
        SystemMessage(content="Extract only the competitor company names from this text. Return ONLY a Python list like: ['Company1', 'Company2', 'Company3']. Nothing else."),
        HumanMessage(content=research_plan)
    ])

    try:
        competitor_names = eval(extraction.content.strip())
    except:
        competitor_names = ["competitor analysis"]

    args_list = [
        (name, company_name, niche, country)
        for name in competitor_names[:5]
    ]

    with ThreadPoolExecutor(max_workers=5) as executor:
        results_list = list(executor.map(search_single_competitor, args_list))

    all_results = [item for sublist in results_list for item in sublist]
    
    # Check if all searches failed due to limit
    all_errors = all(
        len(r) == 1 and r[0].get("error") == "search_limit_reached" 
        for r in results_list
    )
    
    if all_errors:
        raise HTTPException(
            status_code=503,
            detail="search_limit_reached"
        )

    return {"search_results": str(all_results)}

def analyst_node(state: State) -> dict:
    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content="You are a research analyst. Synthesize the search results into clear findings."),
        HumanMessage(content=f"Research plan:\n{state.get('research_plan', '')}\n\nSearch results:\n{state.get('search_results', '')}")
    ])
    return {"final_report": response.content}

def writer_node(state: State) -> dict:
    llm = get_llm()
    company_name = state.get("company_name", "the company")
    country = state.get("country", "United States")
    niche = state.get("niche", "")
    competitor_level = state.get("competitor_level", "All sizes")

    response = llm.invoke([
        SystemMessage(content=f"""You are a professional competitive intelligence writer.
        Write a comprehensive competitor analysis report for {company_name} in the {niche} niche, 
        focused on the {country} market.
        Competitor level analyzed: {competitor_level}
        
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
    competitor_level: str = "All sizes"

class ResearchResponse(BaseModel):
    company_name: str
    research_plan: str
    final_report: str

report_cache = {}
request_tracker = {"date": date.today(), "count": 0}
DAILY_LIMIT = 25

@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {"status": "Competitor Intelligence Agent is running"}


@app.post("/analyze", response_model=ResearchResponse)
def run_research(request: AnalyzeRequest):
    if request_tracker["date"] != date.today():
        request_tracker["date"] = date.today()
        request_tracker["count"] = 0

    if request_tracker["count"] >= DAILY_LIMIT:
        raise HTTPException(
            status_code=429,
            detail="daily_limit_reached"
        )

    cache_key = f"{request.company_name.lower()}_{request.niche.lower()}_{request.country.lower()}_{request.competitor_level.lower()}"
    if cache_key in report_cache:
        return report_cache[cache_key]

    request_tracker["count"] += 1

    config = {"configurable": {"thread_id": f"analyze-{request.company_name[:20]}"}}

    full_query = f"Company: {request.company_name}, Niche: {request.niche}, Country: {request.country}, Competitor Level: {request.competitor_level}"

    initial_state = {
        "messages": [HumanMessage(content=full_query)],
        "company_name": request.company_name,
        "niche": request.niche,
        "country": request.country,
        "competitor_level": request.competitor_level,
        "research_plan": "",
        "search_results": "",
        "final_report": ""
    }

    result = graph.invoke(initial_state, config=config)

    response = ResearchResponse(
        company_name=result.get("company_name", request.company_name),
        research_plan=result["research_plan"],
        final_report=result["final_report"]
    )

    report_cache[cache_key] = response
    return response