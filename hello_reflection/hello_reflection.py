# pip install langchain-ollama==0.2.0 IPython==8.29.0
from langchain_ollama import OllamaLLM
from IPython.display import display_markdown

llm = OllamaLLM(model="llama3.2", temperature=0)

generation_chat_history = [
   {
       "role": "system",
       "content": "You are an experienced Python programmer who generate high quality Python code for users with there explanations"
       "Here's your task: You will Generate the best content for the user's request and give explanation of code line by line. If the user provides critique,"
       "respond with a revised version of your previous attempt."
       "also in the end always ask - Do you have any feedback or would you like me to revise anything?"
       "In each output you will tell me whats new you have added for the user in comparison to earlier output"
   }
]

generation_chat_history.append(
   {
       "role": "user",
       "content": "Generate a Python implementation of the Fibonacci series for beginner students"
   }
)

fibonacci_code = llm.invoke(generation_chat_history)

generation_chat_history.append(
   {
       "role": "assistant",
       "content": fibonacci_code
   }
)

print("Generation Step")
display_markdown(fibonacci_code, raw=True)

#### Reflection Step ####
print("\n\nReflection Step")

reflection_chat_history = [
   {
   "role": "system",
   "content": "You are Nitika Sharma, an experienced Python coder. With this experience in Python generate critique and recommendations for user output on the given prompt",
   }
]
reflection_chat_history.append(
   {
       "role": "user",
       "content": fibonacci_code
   }
)
critique = llm.invoke(reflection_chat_history)
display_markdown(critique, raw=True)

#### Generation Step (2nd Iteration) ####
print("\n\nGeneration Step (2nd Iteration)")

Generation_2 = llm.invoke(generation_chat_history)
display_markdown(Generation_2, raw=True)

#### Reflection Step (2nd Iteration) ####
print("\n\nReflection Step (2nd Iteration)")
reflection_chat_history.append(
   {
       "role": "user",
       "content": Generation_2
   }
)
critique_1 = llm.invoke(reflection_chat_history)
display_markdown(critique_1, raw=True)

#### Generation Step (3rd Iteration) ####
print("\n\nGeneration Step (3rd Iteration)")

generation_chat_history.append(
   {
       "role": "user",
       "content": critique_1
   }
)
Generation_3 = llm.invoke(generation_chat_history)
display_markdown(Generation_3, raw=True)