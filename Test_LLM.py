import openai

def generate_text(prompt, model="gpt-4"):

response = openai.ChatCompletion.create(

     model=model,

     messages=[{"role": "user", "content": prompt}]

)

return response["choices"][0]["message"]["content"]

prompt = "Write a short paragraph about the benefits of AI in healthcare."

print(generate_text(prompt))