# ruff: noqa
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pydantic import BaseModel, Field

import google.auth
from google.auth.exceptions import DefaultCredentialsError

# Environment configuration: support both Local Dev API Key and Cloud Vertex AI
if os.environ.get("GEMINI_API_KEY"):
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"
else:
    try:
        _, project_id = google.auth.default()
        os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
        os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
    except DefaultCredentialsError:
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"

from google.adk.agents import Agent, LlmAgent
from google.adk.agents.context import Context
from google.adk.apps import App
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.models import Gemini
from google.adk.workflow import Workflow
from google.genai import types


# 1. Pydantic model for structured classification
class ClassificationResult(BaseModel):
    is_shipping_related: bool = Field(
        description="True if the user's query is related to shipping (rates, tracking, delivery status, returns, policies). False if unrelated."
    )
    reason: str = Field(
        description="Brief explanation for why the query was classified as shipping-related or unrelated."
    )


# 2. Nodes: functions & LLM agents
def save_query(node_input: types.Content) -> Event:
    """Extracts raw text query from user input and saves it to workflow state."""
    query = ""
    if node_input and node_input.parts:
        query = node_input.parts[0].text
    return Event(output=query, actions=EventActions(state_delta={"user_query": query}))


classifier = LlmAgent(
    name="classifier",
    model=Gemini(
        model="gemini-2.5-flash",
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=(
        "You are an AI router. Classify if the query is related to shipping "
        "(rates, tracking, delivery status, returns, packaging, shipping policies) or unrelated to shipping."
    ),
    output_schema=ClassificationResult,
)


def router(ctx: Context, node_input: dict, user_query: str) -> Event:
    """Routes query based on classifier result, passing user query downstream."""
    is_shipping = node_input.get("is_shipping_related", False)
    route = "shipping" if is_shipping else "unrelated"
    return Event(actions=EventActions(route=route), output=user_query)


faq_agent = LlmAgent(
    name="faq_agent",
    model=Gemini(
        model="gemini-2.5-flash",
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=(
        "You are a helpful customer support representative for a shipping company. "
        "Answer questions about shipping rates, tracking, delivery, or returns. Be polite and concise."
    ),
)


def decline_node(node_input: str) -> Event:
    """Politely declines to answer non-shipping inquiries."""
    msg = (
        "I apologize, but I can only assist with shipping-related inquiries "
        "such as shipping rates, shipment tracking, package delivery, and returns. "
        "How can I help you with your shipping needs today?"
    )
    yield Event(
        content=types.Content(role="model", parts=[types.Part.from_text(text=msg)])
    )
    yield Event(output=msg)


# 3. Define the Graph Workflow
edges = [
    ("START", save_query),
    (save_query, classifier),
    (classifier, router),
    (router, faq_agent, "shipping"),
    (router, decline_node, "unrelated"),
]

root_agent = Workflow(
    name="customer-support-workflow",
    edges=edges,
)

app = App(
    root_agent=root_agent,
    name="app",
)
