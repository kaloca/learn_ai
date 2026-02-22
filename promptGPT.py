from Transformer_basics import TrainGPT

gpt = TrainGPT()
gpt.parse_book("lotr.txt")
gpt.build_vocab()
gpt.tokenize_book()
gpt.init_model(T=256, n_steps=50)
gpt.load("model.pt")
gpt.train()

print("Model ready. Type a prompt (Ctrl+C to exit)\n")

while True:
    try:
        prompt = input(">>> ")

        if prompt.strip() == "":
            continue

        # generate response (adjust args if needed)
        output = gpt.prompt(prompt)

        print(output)
        print()

    except KeyboardInterrupt:
        print("\nExiting.")
        break
