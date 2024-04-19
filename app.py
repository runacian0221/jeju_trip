# 환경 변수에서 API 키 가져오기
import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# CrewAI 라이브러리에서 필요한 클래스 가져오기
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
import gradio as gr

# Web Search Tool
from langchain_community.tools.tavily_search import TavilySearchResults

# PDF Search Tool
from crewai_tools import PDFSearchTool

# Custom Tool
from crewai_tools import tool

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time

# LLM
llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0, api_key=OPENAI_API_KEY)

search_tool = TavilySearchResults(api_key=TAVILY_API_KEY)

pdf_tool = PDFSearchTool(pdf='제주_코스별_여행지.pdf')


@tool("search_place_info")
def search_place_info(place_name: str) -> str:
    """
    Searches for a location on Google Maps.
    Returns operating hours, address, phone number, and fees.
    """


    # Chrome 드라이버 설정
    options = webdriver.ChromeOptions()
    options.add_argument('--headless') 

    service = Service(ChromeDriverManager().install())

    driver = webdriver.Chrome(service=service, options=options)
    driver.get("https://www.google.com/maps/")
    driver.implicitly_wait(10)


    # 검색창에 입력하기
    input_search = driver.find_element(By.ID, 'searchboxinput')
    input_search.send_keys(place_name)
    driver.implicitly_wait(5)
    input_search.send_keys(Keys.RETURN)
    driver.implicitly_wait(5)
    time.sleep(3)


    # 장소가 여러 개 검색된 경우
    if f"{place_name}에대한검색결과" in driver.page_source.replace(" ", ""):
        search_results = driver.find_elements(By.CSS_SELECTOR, 'div > div > div > div > a')

        for n, item in enumerate(search_results):

            try:

                if place_name in item.get_attribute('aria-label'):
                    item.click()
                    driver.implicitly_wait(5)
                    time.sleep(3)
                    break

            except:
                pass


    # 장소 정보 가져오기
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    place_info = soup.find_all('div', attrs={'aria-label': place_name})
    place_info_text = "\n".join([info.text for info in place_info])

    driver.quit()

    return place_info_text

def run_jeju_trip_crew(message):
    # Agent
    jeju_tour_planning_expert = Agent(
    role='Jeju Tour Planning Expert',
    goal='Select the best locations within Jeju based on weather, season, prices, and tourist preferences',
    backstory='An expert in analyzing local data to pick ideal destinations within Jeju Island',
    verbose=True,
    tools=[search_tool, pdf_tool, search_place_info],
    allow_delegation=False,
    llm=llm,
    max_iter=3,
    max_rpm=10,
    )

    jeju_local_expert = Agent(
    role='Jeju Local Expert',
    goal='Provide the BEST insights about the selected locations in Jeju',
    backstory="""A knowledgeable local guide with extensive information
    about Jeju's attractions, customs, and hidden gems""",
    verbose=True,
    tools=[search_tool, pdf_tool, search_place_info],
    allow_delegation=False,
    llm=llm,
    max_iter=3,
    max_rpm=10,
    )



    jeju_travel_concierge = Agent(
    role='Jeju Custom Travel Concierge',
    goal="""Create the most amazing travel itineraries for Jeju including budget and packing suggestions""",
    backstory="""Specialist in Jeju travel planning and logistics with 
    extensive experience""",
    verbose=True,
    ls=[search_tool],
    allow_delegation=False,
    llm=llm,
    )

    # Tasks
    jeju_location_selection_task = Task(
        description='Identify the best locations within Jeju for visiting based on current weather, season, prices, and tourist preferences.',
        agent=jeju_tour_planning_expert,
        expected_output='A list of recommended locations in Jeju, including reasons for each selection'
    )


    jeju_local_insights_task = Task(
        description='Provide detailed insights and information about selected locations in Jeju, including attractions, customs, and hidden gems.',
        agent=jeju_local_expert,
        expected_output='Comprehensive information about each location, including what to see, do, and eat'
    )

    jeju_travel_itinerary_task = Task(
        description='Create a detailed travel itinerary for Jeju that includes budgeting, packing suggestions, accommodations, and transportation.',
        agent=jeju_travel_concierge,
        expected_output='Your Jeju trip plan - including a daily itinerary, estimated budget, and packing list - is all in Korean. Key locations and place names should be provided in Korean, such as "Museum (Art Museum)".".'
    )

    # Crew 생성  

    trip_crew = Crew(
        agents=[jeju_tour_planning_expert, jeju_local_expert, jeju_travel_concierge],
        tasks=[jeju_location_selection_task, jeju_local_insights_task, jeju_travel_itinerary_task],
        process=Process.hierarchical,
        manager_llm=ChatOpenAI(model="gpt-3.5-turbo-0125")   # 비용 과금에 유의 (GPT-4는 비용이 높음). gpt-3.5-turbo로 변경 가능
    )

    #`kickoff` method starts the crew's process
    result = trip_crew.kickoff()

    return result

def process_query(message, history):
    return run_jeju_trip_crew(message)


if __name__ == '__main__':
    app = gr.ChatInterface(
        fn=process_query,
        title="Jeju trip Advisor Bot",
        description="제주여행 관련 트렌드를 파악하여 관광지추천 인사이트를 제공해 드립니다."
    )

    app.launch()
