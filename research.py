from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool

load_dotenv()

class ResearchResponse(BaseModel):
  topic: str
  summary: str
  sources: list[str]
  tools_used: list[str] 

llm = ChatOpenAI(model='gpt-4o-mini')
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
  [
    (
      "system",
      """
      You are a research assiatant that will help generate a research paper.
      Answer the user query and use neccessary tools.
      Wrap the output in this format and provide no other text\n{format_instructions}
      """,
    ),
    ("placeholder", "{chat_history}"),
    ("human", "{query}"),
    ("placeholder", "{agent_scratchpad}"),
  ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wiki_tool, save_tool]
agent = create_tool_calling_agent(
  llm=llm,
  prompt=prompt,
  tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=[])
query = input('What can I help you to research? ')
raw_response = agent_executor.invoke({query})

try: 
  structured_response = parser.parse(raw_response.get('ouput')[0]['text'])
  print(structured_response.topic)
except Exception as e:
  print('Error parsing response', e, "Raw Response - ", raw_response)
