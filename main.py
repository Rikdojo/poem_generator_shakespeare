# app.py

from mlx_lm import load, generate
import gradio as gr


MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
ADAPTER_PATH = "adapters_shakespeare_t8"  

print("Loading model...")
model, tokenizer = load(
    MODEL_NAME,
    adapter_path=ADAPTER_PATH,  #
)
print("Model loaded.")


def generate_poem(title: str) -> str:
    if not title.strip():
        return "Please enter a title."

   
    user_prompt = f"Write a short poem based on the title:\n\"{title}\""

   
    messages = [
        {"role": "user", "content": user_prompt}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
    )


    text = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=200,
    )


 
    return text.strip()

#gradion UI function 
with gr.Blocks() as demo:
    gr.Markdown("#  Shakespeare-style Poem Generator")
    gr.Markdown("Write a poem title, it will generate a short poem with Shakespeare-style ")

    title_box = gr.Textbox(
        label="Title",
        placeholder="e.g. The Moon over the Silent Sea",
    )

    output_box = gr.Textbox(
        label="Generated Poem",
        lines=12,
    )

    generate_button = gr.Button("Generate Poem ✨")

    generate_button.click(
        fn=generate_poem,
        inputs=title_box,
        outputs=output_box,
    )


if __name__ == "__main__":
    demo.launch()
