"""
In PowerShell, use:
	$env:OPENAI_API_KEY="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"
Ensure that the environment variable is set before running this script.
"""

from langchain_openai import ChatOpenAI
from browser_use import Agent
import asyncio

# Initialize the model
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
)

# The Agent
async def main():
    # Create agent with the model
    agent = Agent(
        task="""
        Open Servus.ca website. 
        Open Calculators. 
        Open Loan Calculator. 
        Select Payments from "I want to calculate my" drop down list. 
        Enter loan amount = $15000, 
        Interest rate = 5%. 
        Click Recalculate.
        """,
        llm=llm
    )
    
    await agent.run(max_steps=30)
    input('Press Enter to continue...')

asyncio.run(main())
