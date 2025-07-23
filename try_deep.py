from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import time 

start = time.time()
model_id = "deepseek-ai/deepseek-llm-7b-chat"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    device_map={"": "cpu"}  # принудительно всё в CPU
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

print("mid_works = " + str(time.time() - start))

prompt = (
    "Сделай краткую историю из следующего предложения:\n"
    "Максим потерял тапок.\n"
)

res = pipe(prompt, max_new_tokens=500, do_sample=False)[0]['generated_text']
summary = res.split("<|assistant|>")[-1].strip()

print("\n📄 Сводка от DeepSeek:\n", summary)

print("finish")
print(time.time() - start)