from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import torch
import pandas as pd

# Load dataset
dataset = load_dataset('derek-thomas/squad-v1.1-t5-question-generation')
df = pd.DataFrame(dataset['train'])

# Define a function to split and concatenate questions in each row
def split_and_concatenate_questions(row):
    questions = row['questions'].split("Question: ")
    questions = [q.strip() for q in questions if q]  # Clean and filter
    return " | ".join(questions)  # Combine questions

# Process dataset
df['Questions'] = df.apply(split_and_concatenate_questions, axis=1)
processed_df = df[['context', 'Questions']]

# Define custom dataset class
class QADataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_text = f"generate questions: {self.data.iloc[index]['context']}"
        target_text = self.data.iloc[index]['Questions']

        # Tokenize input and target text
        input_encodings = self.tokenizer(
            input_text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )
        target_encodings = self.tokenizer(
            target_text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )

        # Adjust labels for T5
        labels = target_encodings["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore padding tokens

        return {
            "input_ids": input_encodings["input_ids"].squeeze(),
            "attention_mask": input_encodings["attention_mask"].squeeze(),
            "labels": labels,
        }

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Split dataset into training and validation
train_size = int(0.8 * len(processed_df))
train_dataset = QADataset(processed_df[:train_size], tokenizer)
eval_dataset = QADataset(processed_df[train_size:], tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,  # Adjust based on GPU memory
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    fp16=True,  # Enable mixed precision for faster training
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# Train the model
trainer.train()

# Save the trained model and tokenizer
trainer.save_model("./fine_tuned_t5")  # Save the fine-tuned model
tokenizer.save_pretrained("./fine_tuned_t5")  # Save the tokenizer
print("Model and tokenizer saved successfully!")
