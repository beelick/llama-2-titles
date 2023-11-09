# %%
import os
import yaml
import gspread
from dotenv import load_dotenv

import torch
import pandas as pd

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)

from huggingface_hub import (
    HfApi,
    HfFolder,
)

from peft import (
    LoraConfig,
    PeftModel,
)

from trl import SFTTrainer
from datasets import Dataset

from langchain import (
    HuggingFacePipeline,
    PromptTemplate,
    LLMChain,
)

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

def get_config():
    with open("config_company.yaml", "r") as f:
       return yaml.safe_load(f)


def load_huggingface_auth():
    load_dotenv()
    hf_token = os.getenv("HUGGINGFACE")
    HfFolder.save_token(hf_token)
    return HfApi()


def generate_prompt(prompt_template, batch):
    return {"text": [prompt_template.generate_prompt(title, label) for title, label in zip(batch["desc"], batch["label"])]}


def clean_output(output, categories):
    for cat in categories:
        if cat in output:
            return cat


class Llama2PromptTemplate:
    def __init__(self, system_message=None):
        self.START_SYS = "<s>[INST] <<SYS>>\n"
        self.END = " </s>"
        self.END_SYS = "\n<</SYS>>\n"
        self.START_Q = "\n"
        self.END_Q = " [/INST]"
        self.DEFAULT_SYS = system_message or "You are a helpful assistant."

    def generate_prompt(self, user_message, model_replies=""):
        system_message = f"{self.START_SYS}{self.DEFAULT_SYS}{self.END_SYS}"
        period = self.END if model_replies else ""
        user_message_block = f"{self.START_Q}{user_message}{self.END_Q} {model_replies}{period}"
        return f"{system_message}{user_message_block}"


class Drive:
    def __init__(self, cfg, file_name="creds/service_account.json"):
        self.creds = Credentials.from_service_account_file(file_name, scopes=cfg["SCOPE"])
        self.service = build("drive", "v3", credentials=self.creds)


class Data:
    def __init__(self):
        self.client = None

    def authorize(self, credentials):
        self.client = gspread.authorize(credentials)

    def load_data(self, spreadsheet_id, sheet_name):
        spreadsheet = self.client.open_by_key(spreadsheet_id)
        worksheet = spreadsheet.worksheet(sheet_name)
        return pd.DataFrame(worksheet.get_all_records())

    @staticmethod
    def load_dataset(cfg, data_class, prompt_template):
        spreadsheet_id = cfg["spreadsheet_id"]
        range_name = cfg["range_name"]
        df = data_class.load_data(spreadsheet_id, range_name)
        dataset = Dataset.from_pandas(df[cfg["columns"]])
        dataset = dataset.map(lambda batch: generate_prompt(prompt_template, batch), batched=True)
        dataset = dataset.train_test_split(**cfg["dataset"])
        return dataset["train"], dataset["test"]


class Trainer:
    def __init__(self, cfg, model, train_dataset, test_dataset):
        self.model = model
        self.cfg = cfg
        self.trainer = SFTTrainer(
            model=model.model,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            peft_config=LoraConfig(**cfg["LoraConfig"]),
            tokenizer=model.tokenizer,
            args=TrainingArguments(**self.cfg["TrainingArguments"]),
            **cfg["SFTTrainer"]
        )

    def train(self):
        self.trainer.train()

    def save_model(self, name):
        self.trainer.model.save_pretrained(name)

    def evaluate(self):
        return self.trainer.evaluate()


class Model:
    def __init__(self, cfg):
        self.model = None
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg["pretrained_model_name_or_path"], **cfg["AutoTokenizer"])
        self.initialize_model()

    def __name__(self):
        return self.cfg["pretrained_model_name_or_path"]

    def initialize_model(self):
        self.cfg["bnb_4bit_compute_dtype"] = getattr(torch, self.cfg["bnb_4bit_compute_dtype"])

        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg["pretrained_model_name_or_path"],
            quantization_config=BitsAndBytesConfig(**self.cfg["BitsAndBytesConfig"]),
            **self.cfg["AutoModelForCausalLM"]
        )
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1

    def reload_model(self):
        self.cfg["reload"]["torch_dtype"] = torch.float16
        self.cfg["reload"]["device_map"] = self.cfg["AutoModelForCausalLM"]["device_map"]
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg["pretrained_model_name_or_path"],
            **self.cfg["reload"]["AutoModelForCausalLM"]
        )
        self.model = PeftModel.from_pretrained(self.model, self.cfg["new_model"])
        self.model = self.model.merge_and_unload()

    def push_to_hub(self):
        self.model.push_to_hub(self.cfg["new_model"], **self.cfg["push_to_hub"]["model"])
        self.tokenizer.push_to_hub(self.cfg["new_model"], **self.cfg["push_to_hub"]["tokenizer"])


def finetune(cfg):
    load_huggingface_auth()
    categories = cfg["categories"]
    system_message = cfg["system_message"].format(categories_str=", ".join([f"'{c}'" for c in categories]))
    prompt_template = Llama2PromptTemplate(system_message)

    drive = Drive(cfg["drive"])
    data = Data()
    data.authorize(drive.creds)

    train_dataset, test_dataset = data.load_dataset(cfg, data, prompt_template)

    model = Model(cfg["model"])

    trainer = Trainer(cfg["trainer"], model, train_dataset, test_dataset)
    trainer.train()
    trainer.save_model(cfg["model"]["new_model"])

    print("Evaluation Results:", trainer.evaluate())

    cfg["pipeline"]["torch_dtype"] = torch.bfloat16
    pipe = pipeline("text-generation",
                    model=model.model,
                    tokenizer=model.tokenizer,
                    **cfg["pipeline"]
                    )

    logging.set_verbosity(logging.CRITICAL)
    llm = HuggingFacePipeline(pipeline=pipe, **cfg["HuggingFacePipeline"])
    desc = "Vice President and Chief Information Officer"
    prompt = prompt_template.generate_prompt("{desc}")
    prompt = PromptTemplate(template=prompt, **cfg["PromptTemplate"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    output = llm_chain.run(desc), categories
    print(output)




def chain():
    cfg = get_config()
    load_huggingface_auth()
    categories = cfg["categories"]
    system_message = cfg["system_message"].format(categories_str=", ".join([f"'{c}'" for c in categories]))
    prompt_template = Llama2PromptTemplate(system_message)
    drive = Drive(cfg["drive"])
    data = Data()
    data.authorize(drive.creds)

    model = "meta-llama/Llama-2-7b-chat-hf"

    tokenizer = AutoTokenizer.from_pretrained(model)

    pipe= pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_length=500,
        do_sample=True,
        top_k=30,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )

    llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={"temperature": 0})
    logging.set_verbosity(logging.CRITICAL)
    prompt = prompt_template.generate_prompt("{desc}")
    prompt = PromptTemplate(template=prompt, **cfg["PromptTemplate"])
    return LLMChain(prompt=prompt, llm=llm)

# %%
llm_chain = chain()

# %%
desc = """MERIT is a clinical trial endpoint service provider specializing in the ophthalmology, respiratory, and oncology therapeutic areas. They partner with cros and pharmaceutical and biotech companies to deliver reliable endpoint services in multi-regional clinical trials."""
output = llm_chain.run(desc)
print(output)