# GPT-2 from Scratch in PyTorch

This project implements a transformer-based GPT-2 model from scratch using PyTorch, trained on the TinyStories dataset. The implementation focuses on understanding core transformer concepts by building all components manually without relying on pre-built modules

---

##  Project Structure

The project contains a main Jupyter Notebook:

GPT2.ipynb: Includes the full implementation of the GPT-2 model, training it on the TinyStories dataset, and generating tex

---

## Features

- Custom tokenizer for text processing
- GPT-2-like architecture implemented using PyTorch
- Supports TinyStories dataset or similar text data
- Training loop with loss reporting
- Text generation using temperature scaling

---

## How to Use

### 1. Requirements

- Python 3.8 or higher
- PyTorch 2.0 or higher
- HuggingFace Tokenizers
- tqdm library

Install required packages:

```bash
pip install torch tokenizers tqdm
```

---

### 2. Directory structure:

GPT-2-Project/
├── data/
│   ├── TinyStories-train.txt     # Training data
│   └── TinyStories-valid.txt     # Validation data
├── models/
│   ├── tokenizer.json            # Trained tokenizer
│   └── gpt2_model.pt             # Model weights
└── GPT2.ipynb                    # Main notebook


---

### 3. Load the dataset

Place the file TinyStoriesV2-GPT4-train.txt in the appropriate path as referenced in the notebook.

---

### 4.  Run the Notebook


Open `GPT2.ipynb` using Jupyter Notebook or Kaggle and run all cells in order.


---

 ### 5. Generate Text

After training, you can generate new text samples by setting a prompt and temperature value.
---

##  Model Details


- **Embedding Layer**: Represents tokens and their positions
- **Transformer Blocks**: Includes multi-head self-attention and feedforward layers
- **Output**: Predicts the next token

---

## Example Usage

```python
context = torch.tensor([tokenizer.encode("Once upon a time")], dtype=torch.long)
output = model.generate(context, max_new_tokens=100, temperature=0.8)
print(tokenizer.decode(output[0].tolist()))
```

---

## Example Output

**Prompt**: "In a magical kingdom"  
**Generated**: "In a magical kingdom, there lived a brave knight who loved to explore the enchanted forest. One day, he found a golden key hidden under a sparkling waterfall..."

---

## Customization


- **Model Size**: Modify d_model, num_layers, and num_heads
- **Dataset**: Replace TinyStories with another dataset
- **Settings**: Adjust epochs, batch size, or optimizer parameters

---

## Limitations

- Uses a smaller GPT-2 model (6 layers and 512 dimensions) for efficiency
- Trains on only 10% of TinyStories (modifiable in code)
- A GPU is recommended for faster training

---



