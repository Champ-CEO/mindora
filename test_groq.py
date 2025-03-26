"""Test script to verify Groq API connection."""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

# Check if API key is available
api_key = os.getenv("GROQ_API_KEY")
if not api_key or api_key == "YOUR KEY HERE":
    print("⚠️ ERROR: GROQ_API_KEY not found or not properly set in .env file")
    print("Please ensure you have a valid Groq API key in your .env file")
    exit(1)

# Try both models
models = [
    "llama-3.3-70b-versatile",  # General tasks model
    "deepseek-r1-distill-llama-70b",  # Complex tasks model
]

print("🔍 Testing Groq API connection...")

for model in models:
    print(f"\n📋 Testing model: {model}")
    try:
        # Initialize the Groq chat model
        chat = ChatGroq(
            model=model,
            temperature=0,
        )

        # Simple test message
        message = [
            HumanMessage(content="Hello, can you respond with 'Groq API is working correctly'?")
        ]

        # Make the API call
        print("  🚀 Sending request...")
        response = chat.invoke(message)

        # Print the response
        print(f"  ✅ Response received: {response.content}")

    except Exception as e:
        print(f"  ❌ Error testing {model}: {str(e)}")

print("\n🔍 Groq API validation complete")
