import os
import asyncio
import aiohttp
import logging
import openai
import streamlit as st
import pandas as pd
import json
import re
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import BaseModel, Field

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from Replit Secrets
AI71_BASE_URL = os.environ['AI71_BASE_URL']
AI71_API_KEY = os.environ['AI71_API_KEY']
NEWS_API_KEY = os.environ['NEWS_API_KEY']

# Initialize AI71 client
client = openai.OpenAI(
    api_key=AI71_API_KEY,
    base_url=AI71_BASE_URL,
)


class NewsInput(BaseModel):
    title: str = Field(..., min_length=5, max_length=200)
    content: str = Field(..., min_length=20)
    output_type: str = Field(..., pattern="^(article|social_media)$")


class NewsAgent:

    def __init__(self, name, role_description, workflow):
        self.name = name
        self.role_description = role_description
        self.workflow = workflow
        self.system_message = f"You are {self.name}. {self.role_description}\n\nYour workflow: {self.workflow}"

    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=4, max=10))
    async def run(self, input_text):
        try:
            response = await asyncio.to_thread(client.chat.completions.create,
                                               model="tiiuae/falcon-180b-chat",
                                               messages=[
                                                   {
                                                       "role":
                                                       "system",
                                                       "content":
                                                       self.system_message
                                                   },
                                                   {
                                                       "role": "user",
                                                       "content": input_text
                                                   },
                                               ])
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in {self.name} agent: {str(e)}")
            raise


class Researcher(NewsAgent):

    def __init__(self):
        super().__init__(
            "Researcher",
            "You are a professional researcher with high-rank institution experience, working in a news startup. You will research news using the News API to get detailed material with enough information about the defined subject.",
            "Receive topics and process details from the editor, generate a concise search query, research using News API, before handing it to the writer."
        )

    async def generate_search_query(self, editor_output):
        prompt = f"""Based on the following editor's output, generate a concise search query (maximum 5 words) for researching news:

{editor_output}

Concise search query:"""

        try:
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model="tiiuae/falcon-180b-chat",
                messages=[
                    {
                        "role":
                        "system",
                        "content":
                        "You are a helpful assistant that generates concise search queries."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    },
                ])
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating search query: {str(e)}")
            return editor_output[:
                                 100]  # Fallback to first 100 chars of editor output if query generation fails

    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=4, max=10))
    async def search_news_api(self, query):
        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}&language=en&sortBy=relevancy&pageSize=5"
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_content = await response.text()
                        logger.error(
                            f"Error: News API returned status code {response.status}. Response: {error_content}"
                        )
                        return f"Error fetching news: Status {response.status}, Message: {error_content}"
            except aiohttp.ClientError as e:
                logger.error(f"Error accessing News API: {str(e)}")
                return f"Error accessing News API: {str(e)}"

    async def run(self, input_text):
        search_query = await self.generate_search_query(input_text)
        logger.info(f"Generated search query: {search_query}")
        news_api_results = await self.search_news_api(search_query)
        if isinstance(news_api_results,
                      str) and news_api_results.startswith("Error"):
            return f"Research failed: {news_api_results}"
        combined_results = f"Search query: {search_query}\n\nNews API results: {news_api_results}"
        return await super().run(combined_results)


class Writer(NewsAgent):

    def __init__(self):
        super().__init__(
            "Writer",
            "You are a top-tier, creative, talented writer with enough experience in enriching developing stories with the exact right backgrounds and drafting elegant pitches.",
            "Receive data and researched topics from the researcher, and write either a detailed 4-paragraph background for articles, or 3 social media posts for Facebook, Twitter, and Instagram."
        )

    async def run(self, input_text, output_type):
        prompt = f"{input_text}\n\nBased on this information, "
        if output_type == 'article':
            prompt += "write a detailed 4-paragraph background for an article."
        else:
            prompt += "write 3 social media posts: one for Facebook, one for Twitter, and one for Instagram."

        return await super().run(prompt)


class Producer(NewsAgent):

    def __init__(self):
        super().__init__(
            "Producer",
            "You are a talented producer with a high graphic imagination and experience in enriching news stories and articles with charts, maps, and graphs.",
            "Suggest three graphs and charts that could be added to the article, implement them by providing JSON structures for simple line, bar, or pie charts, then hand it to the editor."
        )

    async def run(self, input_text):
        prompt = f"""{input_text}

Based on this information:
1. Suggest three graphs or charts that could be added to the article.
2. Implement each of them by providing a JSON structure for a simple line, bar, or pie chart. Use the following format for each graph:

GRAPH 1:
{{
    "type": "line",  // or "bar" or "pie"
    "data": {{
        "labels": ["Label1", "Label2", "Label3"],
        "datasets": [
            {{
                "label": "Dataset Label",
                "data": [value1, value2, value3]
            }}
        ]
    }}
}}

GRAPH 2:
(Second JSON structure here)

GRAPH 3:
(Third JSON structure here)

Make sure to format each JSON structure as a valid JSON object, without any additional text or explanations within the JSON itself. Separate each graph with the GRAPH 1, GRAPH 2, GRAPH 3 labels.

Respond with:
SUGGESTIONS:
(Your three suggestions here)

IMPLEMENTED GRAPHS:
(Your three JSON structures here, separated by the GRAPH labels)
"""
        return await super().run(prompt)


class Publisher(NewsAgent):

    def __init__(self):
        super().__init__(
            "Publisher",
            "You are an experienced, fast-editing journalist with good tech knowledge, responsible for final publishing to ensure the material is aligned with news conditions.",
            "Receive the material from the editor, ensure it's ready for publishing, and publish it in the appropriate format."
        )

    async def run(self, input_text, output_type):
        if output_type == 'article':
            prompt = f"""{input_text}

Based on this information, provide the following:

1. HEADLINES:
   Suggest 3 catchy headlines for the article.

2. BACKGROUND:
   Provide a concise background and key information for the article.

3. CHART IDEAS:
   Suggest 3 chart ideas that could enhance the article.

4. IMPLEMENTED CHART:
   Implement one of the chart ideas as a JSON structure for a simple line, bar, or pie chart.

Format your response exactly as follows:

HEADLINES:
1. [Headline 1]
2. [Headline 2]
3. [Headline 3]

BACKGROUND:
[Concise background and key information]

CHART IDEAS:
1. [Chart idea 1]
2. [Chart idea 2]
3. [Chart idea 3]

IMPLEMENTED CHART:
[JSON structure for the implemented chart]
"""
        else:  # social_media
            prompt = f"""{input_text}

Based on this information, create the following social media posts:

1. FACEBOOK POST:
   A detailed post suitable for Facebook (1-2 paragraphs).

2. TWITTER POST:
   A concise post suitable for Twitter (max 280 characters).

3. INSTAGRAM POST:
   A visually descriptive post suitable for Instagram, along with a suggestion for an accompanying image or graph.

4. IMPLEMENTED CHART:
   Create a JSON structure for a simple line, bar, or pie chart that could accompany the Instagram post.

Format your response exactly as follows:

FACEBOOK POST:
[Facebook post content]

TWITTER POST:
[Twitter post content]

INSTAGRAM POST:
[Instagram post content]
[Image/graph suggestion]

IMPLEMENTED CHART:
[JSON structure for the implemented chart]
"""
        return await super().run(prompt)


# Initialize agents
lead_editor = NewsAgent(
    "Lead Editor",
    "You are a lead editor in a well-known and tier one news outlet. Your work is to receive news leads from journalists, decide how to process the news, what backgrounds and additions it will need, and suggest the necessary statements required from guests to enrich the story.",
    "Receive the story from the journalist, process it, hand it to the team, then receive it from the producer before sending it to the publisher."
)

researcher = Researcher()
writer = Writer()
producer = Producer()
publisher = Publisher()


async def process_news(news_input: NewsInput):
    try:
        editor_output = await lead_editor.run(
            f"{news_input.title}\n\n{news_input.content}")
        researcher_output = await researcher.run(editor_output)

        if researcher_output.startswith("Research failed:"):
            return f"News processing failed due to research error: {researcher_output}"

        writer_output = await writer.run(researcher_output,
                                         news_input.output_type)
        producer_output = await producer.run(writer_output)
        final_output = await publisher.run(producer_output,
                                           news_input.output_type)
        return final_output
    except Exception as e:
        logger.error(f"Error in news processing pipeline: {str(e)}")
        return f"News processing failed: {str(e)}"


def find_json_objects(text):
    """Find all JSON objects in the given text."""
    pattern = r'\{[^{}]*\}'
    potential_jsons = re.findall(pattern, text)

    valid_jsons = []
    for js in potential_jsons:
        try:
            json.loads(js)
            valid_jsons.append(js)
        except json.JSONDecodeError:
            pass

    return valid_jsons


# Streamlit interface
st.title("AI News Production System")

# Predefined suggestions
suggestions = [
    "US elections candidates for 2024",
    "Oil market trends in second quarter of 2024",
    "Food prices trends in the world for first quarter of 2024"
]

# Create a radio button for topic selection
topic_selection = st.radio("Choose a topic or enter your own:",
                           ["Custom"] + suggestions)

if topic_selection == "Custom":
    topic = st.text_input("Enter a news topic to research:", "")
else:
    topic = topic_selection

output_type = st.selectbox("Choose output type:", ["article", "social_media"])

# ... (previous code remains the same)

# Update this part in your Streamlit interface
if st.button("Generate News"):
    if topic:
        with st.spinner("Generating news..."):
            news_input = NewsInput(
                title=topic,
                content=f"Research and write about: {topic}",
                output_type=output_type)
            result = asyncio.run(process_news(news_input))

            st.subheader("Raw Output")
            st.text(result)  # Display raw output for debugging

            if result:
                if result.startswith("News processing failed"):
                    st.error(result)
                else:
                    st.success("News processed successfully!")

                    # Split the result into sections
                    sections = result.split('\n\n')

                    for section in sections:
                        if ':' in section:
                            header, content = section.split(':', 1)
                            st.subheader(header.strip())
                            st.write(content.strip())

                    # Find all JSON objects in the result
                    json_objects = find_json_objects(result)

                    if json_objects:
                        st.subheader("Implemented Chart")
                        for i, json_str in enumerate(json_objects):
                            st.write(f"JSON Object {i+1}:")
                            st.code(json_str)  # Display raw JSON string
                            try:
                                graph_data = json.loads(json_str)

                                # Debug: Print the structure of graph_data
                                st.write("Debug: Graph Data Structure")
                                st.json(graph_data)

                                if not isinstance(graph_data, dict):
                                    st.warning(
                                        f"Unexpected data type: {type(graph_data)}. Expected a dictionary."
                                    )
                                    continue

                                # Attempt to display data in a flexible manner
                                st.write("Data Display:")
                                if 'data' in graph_data:
                                    data = graph_data['data']
                                    if isinstance(data, dict):
                                        for key, value in data.items():
                                            st.write(f"{key}: {value}")
                                    elif isinstance(data, list):
                                        st.table(data)
                                    else:
                                        st.write(data)
                                else:
                                    st.table(graph_data)

                                # Attempt to create a chart if possible
                                if 'data' in graph_data and 'labels' in graph_data[
                                        'data'] and 'datasets' in graph_data[
                                            'data']:
                                    labels = graph_data['data']['labels']
                                    values = graph_data['data']['datasets'][0][
                                        'data']
                                    df = pd.DataFrame({
                                        'x': labels,
                                        'y': values
                                    })
                                    st.line_chart(df.set_index('x'))
                                    st.bar_chart(df.set_index('x'))

                            except json.JSONDecodeError:
                                st.warning(
                                    f"Failed to parse graph data: {json_str}")
                            except Exception as e:
                                st.error(
                                    f"Error processing graph data: {str(e)}")
                    else:
                        st.info("No graph data found in the output.")
            else:
                st.error("Failed to process news. Please try again.")
    else:
        st.warning(
            "Please select a topic or enter your own before generating news.")
