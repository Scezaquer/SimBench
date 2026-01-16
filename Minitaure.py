from unsloth import FastLanguageModel
import transformers

model, tokenizer = FastLanguageModel.from_pretrained(
  model_name = "marcelbinz/Llama-3.1-Minitaur-8B",
  max_seq_length = 32768,
  dtype = None,
  load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

pipe = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            trust_remote_code=True,
            pad_token_id=0,
            do_sample=True,
            temperature=1.0,
            max_new_tokens=128,
)

prompt = "You are in classical trolley problem scenario. A runaway trolley is heading down the tracks towards five people tied up and unable to move. You have the ability to pull a lever to divert the trolley onto another track, where there is one person tied up. What do you do? Explain your reasoning step by step.\n\nAnswer:"

print(prompt)

choice = pipe(prompt)[0]['generated_text'][len(prompt):]
print(choice)