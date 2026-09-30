from ericai import EricAI
 
client = EricAI()
resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-32B-Instruct",  # tool-calling / structured output
    messages=[{"role": "user", "content": "Say hello"}],
    max_tokens=200,
    temperature=0.3,
)
print(resp.choices[0].message.content)