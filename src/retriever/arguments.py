from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "temperature for softmax"}
    )


@dataclass
class DataArguments:
    dataset_name: str = field(
        default='json', metadata={"help": "huggingface dataset name"}
    )

    dataset_config: str = field(
        default=None, metadata={"help": "huggingface dataset config, useful for datasets with sub-datasets"}
    )

    dataset_path: str = field(
        default=None, metadata={"help": "Path to local data files or directory"}
    )

    dataset_split: str = field(
        default='train', metadata={"help": "dataset split"}
    )

    dataset_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the data downloaded from huggingface"}
    )

    dataset_number_of_shards: int = field(
        default=1, metadata={"help": "number of shards to split the dataset into"}
    )

    dataset_shard_index: int = field(
        default=0, metadata={"help": "shard index to use, to be used with dataset_number_of_shards"}
    )

    train_group_size: int = field(
        default=8, metadata={"help": "number of passages used to train for each query"}
    )

    encode_is_query: bool = field(default=False)
    encode_output_path: str = field(default=None, metadata={"help": "where to save the encode"})

    query_max_len: Optional[int] = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for query. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    passage_max_len: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    pad_to_multiple_of: Optional[int] = field(
        default=16,
        metadata={
            "help": "If set will pad the sequence to a multiple of the provided value. This is especially useful to "
                    "enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta)."
        },
    )


@dataclass
class TevatronTrainingArguments(TrainingArguments):
    warmup_ratio: float = field(default=0.1)