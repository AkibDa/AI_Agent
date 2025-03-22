import google.generativeai as genai
from key import API_KEY

genai.configure(api_key=API_KEY)

model = genai.GenerativeModel("gemini-2.0-flash")


while True:
  
  prompt = input("Enter your prompt: ")
  
  if(prompt == "exit"):
    break
  else:
    response = model.generate_content(prompt)
    print(response.text)


