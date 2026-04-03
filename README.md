# Competitor Intelligence Agent

A **production-grade** multi-agent system that performs deep competitive intelligence research on any company.

Built with **LangGraph + CrewAI** for reliable agent orchestration, **Groq** for fast LLM inference, and **SerpAPI** for real-time web search.

## ✨ Features

- Autonomous Research Planning — Identifies top 4-5 competitors in a specific country/niche
- Multi-Agent Workflow — Planner → Search → Analyst → Writer
- Production Ready — Proper state management, memory, guardrails, and error handling from day one
- Clean Architecture — FastAPI backend with structured output
- Ready for Live Streaming — Designed for real-time frontend integration

## Tech Stack

- **Backend**: FastAPI + LangGraph + CrewAI
- **LLM**: Groq (Llama-3.3-70B-Versatile)
- **Search Engine**: SerpAPI (Google Search)
- **State Management**: LangGraph with MemorySaver

## API Endpoints

- `GET /` → Health check
- `POST /analyze` → Run full competitor intelligence report

**Example Request:**
```json
{
  "company_name": "Nike",
  "niche": "Sportswear",
  "country": "United States"
}

How to Run Locally

# Clone the repository
git clone https://github.com/MuneebAhmed25211/competitor-intelligence-agent.git

# Go into the project
cd competitor-intelligence-agent

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file and add your keys
copy .env.example .env
# Add GROQ_API_KEY and SERPAPI_API_KEY in .env file

# Run the server
uvicorn main:app --reload