# poem_generator_shakespeare
Generate Shakespeare-like poems using a fine-tuned Llama-3.2-1B-Instruct model.

This project fine-tunes Llama-3.2-1B-Instruct with Shakespeare’s sonnets using LoRA on Apple’s MLX framework
and provides a simple Gradio web app to generate poems from a user-given title.

# Installation
pip install --upgrade pip 
pip install mlx mlx-lm gradio scikit-learn
