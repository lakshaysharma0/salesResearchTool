# imports
import re
from firecrawl import FirecrawlApp
from tavily import TavilyClient
import yfinance as yf
from langsmith import Client
import os
import operator
from typing import TypedDict, Annotated, List, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from datetime import datetime
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import os
import certifi
from firecrawl import Firecrawl
from sec_api import ExtractorApi, QueryApi
import praw

os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
load_dotenv()
# Initialize Clients
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
firecrawl_app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
sec_query_api = QueryApi(api_key=os.getenv("SEC_API_KEY"))
sec_extractor_api = ExtractorApi(api_key=os.getenv("SEC_API_KEY"))

# models
evaluator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
planner_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
summarize_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
final_research_model_llm = ChatOpenAI(model="gpt-5.4", temperature=0.7)

print("🚀 Cloud Brains (GPT-4o mini) are online.")
print("✅ Environment ready.")
print(f"Tavily Key: {'Found' if os.getenv('TAVILY_API_KEY') else 'Missing'}")
print(
    f"FireCrawl Key: {'Found' if os.getenv('FIRECRAWL_API_KEY') else 'Missing'}")
print(f"FMP Key: {'Found' if os.getenv('FMP_API_KEY') else 'Missing'}")

client = Client()
try:
    # Try to fetch your projects
    projects = list(client.list_projects())
    print(f"✅ Success! Connection active. Found {len(projects)} projects.")
except Exception as e:
    print(f"❌ Connection Error: {e}")

# state Defining


class researchState(TypedDict):
    my_company_profile: str
    company_name: str
    website_url: str

    search_queries: Annotated[List[str], operator.add]
    new_search_query: str
    new_search_answer: str
    search_history: Annotated[List[str], operator.add]
    iteration: int
    max_iteration: int
    evaluation: str

    research_piles: Annotated[List[str], operator.add]

    full_report: str

# structure output for critic


class critic_output(BaseModel):
    evaluation: Literal["yes", "no"] = Field(
        ..., description="if ssearch query and answer is correct or not")


structured_evaluator_llm = evaluator_llm.with_structured_output(critic_output)

# main research node loop


def search_query_gen(state: researchState):
    current_date = datetime.now().strftime("%B %Y")
    message = [
        SystemMessage(
            content=f"""you are a research Assistant and your work is to generate a optimized search query for a search engine, This is the current date: "{current_date}" """),
        HumanMessage(content=f"""
create a new search query for my market research on the company: "{state['company_name']}".
These are the previous search queries: "{state['search_queries']}".
This is the context that I allready have: "{state['search_history']}"
RULES:
-ALLWAYS MAKE A QUERY WHOSE SEARCH WILL ADD TO THE CONTEXTED INFORMATION.
-NEVER MAKE A SAME QUERY AGAIN THAT IS ALLREADY PRESENT IN THE PREVIOUS QUERIES, IF PRESENT THEN SHIFT THE FOCUS OF QUERY TO A NEW TOPIC THAT CAN HELP RELATED TO THE MARKET RESEARCH.
-ALLWAYS USE THIS CURRENT DATE: {current_date} PROVIDED AS THE TODAYS DATE AND MAKE QUERYS BASED ON THAT DATE ONLY AND AROUND THAT TIME ONLY.
-ALLWAYS RESPOND IN A ONE LINE SEARCH QUERY.
""")
    ]
    search_query = planner_llm.invoke(message).content

    return {'new_search_query': search_query, 'search_queries': [search_query]}


def research_search_node(state: researchState):
    latest_query = state["new_search_query"]
    print(f"--- 📰 Node: News Hunter (Searching: {latest_query}) ---")
    search = tavily.search(query=latest_query, topic="news", max_results=3)
    context = ""
    for r in search['results']:
        context += f"\nSOURCE: {r['title']}\nCONTENT: {r['content'][:2500]}\n---"

    return {'new_search_answer': context}


def critic(state: researchState):
    current_date = datetime.now().strftime("%B %Y")

    message = [
        SystemMessage(
            content="You are a research Assistant and your work is to check and tell if the search query is somewhat getting answered by the search answer."),
        HumanMessage(content=f"""
The Search query is: "{state['new_search_query']}"
The Search result is : "{state['new_search_answer']}"
If the query is getting answered then answer "yes" otherwise "no" 
RULES:
- Be a little lenient in evaluating.
- Always follow the "yes" or "no" answer format.
""")
    ]

    result = structured_evaluator_llm.invoke(message)
    evaluation_answer = result.evaluation
    new_iteration = state["iteration"] + 1
    updates = {
        "evaluation": evaluation_answer,
        "iteration": new_iteration
    }

    if evaluation_answer == "yes":
        message2 = [
            SystemMessage(
                content="you are a summarizer, that will summarize the context given in 1000-1200 words max"),
            HumanMessage(content=f"""
the content to summarize is this: "{state['new_search_answer']}"
RULES:
-BE LEANIENT IN SUMMARIZING THAT MEANS THAT ALL THE INFORMATION SHOULD BE THERE.
            """)
        ]

        summarized_content = summarize_llm.invoke(message2).content
        # Again, operator.add handles the appending. Just pass the new item in a list.
        updates["search_history"] = [summarized_content]

    return updates

# simple nodes


def sec_intel_node(state: researchState):
    company = state['company_name']
    print(f"--- 📄 Node: SEC Intelligence (Deep Dive: {company}) ---")

    # 1. Ticker Mapping (Ensures we use 'TSLA' instead of 'Tesla')
    name_to_ticker = {"Tesla": "TSLA", "Apple": "AAPL", "Nvidia": "NVDA"}
    ticker = name_to_ticker.get(company, company).upper()

    try:
        # 2. Query for the latest 10-K (Excluding amendments for full text)
        query = {
            "query": f"ticker:{ticker} AND formType:\"10-K\" AND NOT formType:\"10-K/A\"",
            "from": "0", "size": "1", "sort": [{"filedAt": {"order": "desc"}}]
        }
        response = sec_query_api.get_filings(query)
        filings = response.get('filings', [])

        if not filings:
            return {"research_piles": [f"SEC DATA: No full 10-K found for {ticker}."]}

        filing_url = filings[0].get('linkToFilingDetails')

        # 3. Extract Item 1A (Risk Factors) - The 83k character block
        raw_risk_factors = sec_extractor_api.get_section(
            filing_url, "1A", "text")

        if not raw_risk_factors or len(raw_risk_factors) < 100:
            return {"research_piles": [f"SEC DATA: Item 1A was empty for {ticker}."]}

        # --- 🛠️ MAP-REDUCE LOGIC ---

        # 4. Map Phase: Split into chunks of ~18,000 characters
        chunk_size = 18000
        chunks = [raw_risk_factors[i:i + chunk_size]
                  for i in range(0, len(raw_risk_factors), chunk_size)]

        partial_summaries = []
        for i, chunk in enumerate(chunks):
            print(f"   💡 Processing SEC Chunk {i+1}/{len(chunks)}...")
            map_prompt = [
                SystemMessage(content="""You are a financial risk analyst. 
                Extract and summarize the critical business risks from this section. 
                Focus on: Production bottlenecks, AI/Technical hurdles, and Competitive threats."""),
                HumanMessage(content=chunk)
            ]
            # Use summarize_llm (temp 0.3) for stability
            summary = summarize_llm.invoke(map_prompt).content
            partial_summaries.append(summary)

        # 5. Reduce Phase: Synthesize into a Master Executive Summary
        print("   🧠 Synthesizing Final SEC Report...")
        reduce_prompt = [
            SystemMessage(content="""You are a Master Sales Strategist. 
            Review these partial summaries of SEC Risk Factors.
            Create a high-impact 'Paper Trail' report that highlights the 3 biggest 'Pain Points'.
            Format it as a concise executive summary for a sales research brief."""),
            HumanMessage(content="\n\n".join(partial_summaries))
        ]

        final_sec_summary = summarize_llm.invoke(reduce_prompt).content

        return {"research_piles": [f"SEC INTELLIGENCE (10-K):\n{final_sec_summary}"]}

    except Exception as e:
        print(f"❌ SEC Node Failed: {e}")
        return {"research_piles": [f"SEC Error for {ticker}: {str(e)}"]}


def finance_node(state: researchState):
    print("----finance Node Running----")
    company = state['company_name']

    name_to_ticker = {"Tesla": "TSLA", "Apple": "AAPL", "Nvidia": "NVDA"}
    ticker_symbol = name_to_ticker.get(company, company)

    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info
        # If yfinance returns an empty dict or error
        if not info.get('regularMarketPrice') and not info.get('currentPrice'):
            return {"research_piles": [f"FINANCIAL SNAPSHOT: Could not retrieve valid ticker data for {ticker_symbol}."]}

        finance_data = f"""
        FINANCIAL SNAPSHOT ({ticker_symbol}):
        - Current Price: ${info.get('currentPrice') or info.get('regularMarketPrice')}
        - Market Cap: {info.get('marketCap')}
        - Profit Margin: {info.get('profitMargins')}
        - Revenue Growth: {info.get('revenueGrowth')}
        """
        return {'research_piles': [finance_data]}
    except Exception as e:
        print(f"❌ yfinance error for {ticker_symbol}: {e}")
        return {'research_piles': [f"FINANCIAL SNAPSHOT: Error retrieving data for {ticker_symbol} -> {str(e)}"]}


def website_scraper(state: researchState):
    url = state['website_url']
    print(f"--- 🕸️ Node: Website Hunter (Scraping {url}) ---")

    try:
        # We'll use the standard scrape_url method
        result = firecrawl_app.scrape(
            url=url,
            formats=['markdown']
        )

        # --- THE FIX IS HERE ---
        # 1. If it's a custom Object (like 'Document')
        if hasattr(result, 'markdown'):
            raw_md = result.markdown
        # 2. If it's a Dictionary (older SDK versions)
        elif isinstance(result, dict):
            raw_md = result.get('markdown', result.get(
                'data', {}).get('markdown', ""))
        # 3. Ultimate Fallback
        else:
            raw_md = str(result)

        # Ensure we don't pass 'None' to regex if the scrape came back empty
        if not raw_md:
            raw_md = ""

        # Strip markdown images ![alt](url) to save LLM tokens
        clean_md = re.sub(r'!\[.*?\]\(.*?\)', '', raw_md)

        return {"research_piles": [f"WEBSITE DATA:\n{clean_md[:1500]}"]}

    except Exception as e:
        print(f"❌ Website scrape failed: {e}")
        return {"research_piles": [f"Website Scrape Error for {url}: {str(e)}"]}


# final report generation nodes
def piling_all_report(state: researchState):
    print("--- 📚 Node: Piling All Research ---")
    history_list = state.get("search_history", [])

    return {"research_piles": history_list}


def generate_final_research(state: researchState):
    print("--- 🧠 Node: Final Synthesis & Sales Pitch ---")

    # Joining the piles with clear headers for the LLM
    compiled_research = "\n\n".join(state.get('research_piles', []))

    prompt = [
        SystemMessage(content="""
        You are a Master Sales Strategist and Elite Executive Consultant. 
        Your style is to understated, highly informed, surgical, and authoritative. 
        You do not use 'hype' or marketing fluff. You use 'The Paper Trail' (SEC data) to find the truth 
        and 'Market Momentum' (News) to find the opportunity.my company is tech mahindra, Your goal is to turn raw data into a high-stakes research report and provide sales pitches for my company.
        """),
        HumanMessage(content=f"""
        PROJECT: Executive Research & Strategic Hooks for {state['company_name']}
        MY COMPANY PORTFOLIO: {state['my_company_profile']}

        INPUT DATA:
        {compiled_research}
        
        TASK:
        Synthesize the raw data into a high-stakes Research Report. 
        The tone must be calm, polished and focused on non-obvious insights.
        
        REPORT STRUCTURE:
        
        1. FINANCIAL HEALTH (The Math)
           - Do not just list numbers. Interpret them. 
           - Contrast the live stock data with the long-term risks found in the SEC 10-K.
        
        2. STRATEGIC MOMENTUM (The Narrative)
           - What is the company's current obsession? 
           - Use recent news to identify their pivot points (e.g., AI clusters, Robotaxis, specific market expansions).
        
        3. THE PAPER TRAIL (Legal & Structural Truth)
           - Explicitly reference the latest SEC Filing (10-K or 10-K/A). 
           - What are they ADMITTING is a problem? (Use the Risk Factors Item 1A).
        
        4. COMPETITIVE INTELLIGENCE (The Gap)
           - Who is actually hurting them? 
           - Identify the 'Gap' where the company is vulnerable and where a partner could provide a 'surgical' solution.
        
        5. THE SURGICAL HOOKS (Sales Entry Points)
           - Provide three opening lines that establish INSTANT authority. 
           - RULE: Each hook must mention a specific, non-obvious detail from the SEC 10-K or a recent 48-hour news event. 
           - Avoid: 'I saw you're doing well.'
           - Use: 'I noticed your April 30th filing mentioned energy cost volatility for AI clusters—I have a strategy for that.'
        
        FORMATTING:
        - Use Markdown headers.
        - Use bolding for 'High-Signal' phrases.
        - Keep it executive-length (approx 800-1000 words).
        """)
    ]

    response = final_research_model_llm.invoke(prompt).content
    return {"full_report": response}

# Conditional edge fuction


def route_search_evaluation(state: researchState):
    print(
        f"--- 🛣️ Routing: Completed {state['iteration']} of {state['max_iteration']} searches ---")
    if state['iteration'] >= state['max_iteration']:
        print("🏁 All search phases complete. Proceeding to Synthesis.")
        return "moving_to_piling"
    print("🔄 Moving to next search phase.")
    return "gen_query_again"

# Building Graph Here


graph = StateGraph(researchState)


graph.add_node("search_query_gen", search_query_gen)
graph.add_node("query_web_search", research_search_node)
graph.add_node("critic", critic)
graph.add_node("finance_info_puller", finance_node)
graph.add_node("web_scraper", website_scraper)
graph.add_node("info_combining", piling_all_report)
graph.add_node("full_research", generate_final_research)
graph.add_node("sec_filing", sec_intel_node)

graph.add_edge(START, 'search_query_gen')
graph.add_edge(START, 'finance_info_puller')
graph.add_edge(START, 'web_scraper')
graph.add_edge(START, 'sec_filing')

graph.add_edge('search_query_gen', 'query_web_search')
graph.add_edge('query_web_search', 'critic')

graph.add_edge('finance_info_puller', END)
graph.add_edge('web_scraper', END)
graph.add_edge('sec_filing', END)

graph.add_conditional_edges(
    'critic',
    route_search_evaluation,
    {
        "moving_to_piling": "info_combining",
        "gen_query_again": "search_query_gen"
    }
)

graph.add_edge('info_combining', 'full_research')

graph.add_edge('full_research', END)

sales_researcher = graph.compile()

initial_state = {
    "my_company_profile": "[Tech Mahindra](https://www.techmahindra.com?utm_source=chatgpt.com), a flagship company of the [Mahindra Group](https://www.mahindra.com?utm_source=chatgpt.com), is one of the world’s leading IT services, consulting, telecom engineering, and digital transformation enterprises, operating across more than 90 countries with a strong presence in industries such as telecom, banking, manufacturing, healthcare, automotive, retail, aerospace, and government sectors. Originally recognized for its telecom and enterprise IT services expertise, the company is now aggressively transforming itself into an AI-first global technology organization focused on enterprise AI execution, autonomous systems, cloud modernization, telecom intelligence, and next-generation digital infrastructure. Under its current strategy, Tech Mahindra is heavily investing in cutting-edge technologies including Generative AI, Agentic AI, Sovereign AI, AI orchestration systems, Large Language Models (LLMs), AI governance, cloud-native infrastructure, quantum computing, cybersecurity, digital twins, geospatial AI, robotics, autonomous operations, and cognitive networks. The company is developing proprietary platforms such as Orion for multi-agent AI orchestration and enterprise workflow automation, IndusLLM for India-focused sovereign AI and multilingual language intelligence, Yantr.ai for manufacturing and operational intelligence, and Kornea for workforce analytics and AI-powered productivity systems. Through its advanced R&D division called Makers Lab, Tech Mahindra is actively researching and building solutions around AI infrastructure, advanced analytics, autonomous telecom systems, quantum technologies, and intelligent enterprise automation. The company has also established strategic partnerships with global technology leaders such as [Microsoft](https://www.microsoft.com?utm_source=chatgpt.com), [Google Cloud](https://cloud.google.com?utm_source=chatgpt.com), and [AMD](https://www.amd.com?utm_source=chatgpt.com) to strengthen its AI ecosystems, cloud transformation capabilities, and enterprise-scale infrastructure modernization initiatives. Tech Mahindra remains one of the strongest telecom-focused IT firms globally, with deep expertise in 5G deployment, Open RAN, virtualized networks, telecom cloud platforms, OSS/BSS modernization, AI-powered network automation, and autonomous network operations. Alongside telecom, the company is expanding rapidly into digital engineering, smart manufacturing, predictive maintenance, smart city infrastructure, geospatial intelligence, digital twin ecosystems, cloud transformation, AI-driven cybersecurity, and industry-specific enterprise AI systems. Its long-term vision is centered around becoming a global leader in AI-powered enterprise transformation by combining deep industry expertise, scalable AI platforms, cloud-native architectures, intelligent automation, and next-generation infrastructure solutions to help enterprises transition from traditional digital systems into fully autonomous, AI-driven operational ecosystems.",
    "company_name": "Tesla",
    "website_url": "https://www.tesla.com",
    "iteration": 0,           # Start at 0
    "max_iteration": 4,       # Stop after 3 searches
    "search_queries": [],
    "search_history": [],
    "research_piles": [],
    "new_search_query": "",
    "new_search_answer": "",
    "evaluation": "",
    "full_report": ""
}

result = sales_researcher.invoke(initial_state)
print(result["full_report"])
