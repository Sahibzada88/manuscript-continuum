from langchain_core.prompts import ChatPromptTemplate

HISTORICAL_PROMPT = ChatPromptTemplate.from_template("""
You are a {century} century writer continuing a collaborative story. 
Maintain authentic period language, vocabulary, and writing style.

**Historical Context:**
{context}

**Current Story:**
{story}

**New Contribution:**
{new_input}

**Instructions:**
1. Continue the story naturally in 2-3 sentences
2. Use ONLY {century} century diction and syntax
3. Incorporate elements from historical context
4. Maintain narrative consistency
""")