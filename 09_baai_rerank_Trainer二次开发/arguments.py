import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help":"模型路径或者名称"}
    )

    config_name: Optional[str] = field(
        default=None, metadata={"help":"预训练配置文件的name 或者 path"}
    )

    tokenizer_name: Optional[str] = field(
        default=None, metadata={'help':'预训练模型的name 或者 path'}
    )

    cache_dir: Optional[str] = field(
        default=None, metadata={'help':'用来存储缓存的路径'}
    )

@dataclass
class DataArguments:
    train_data: str = field(
        default=None, metadata={'help':'训练数据的路径'}
    )
    train_group_size: int = field(default=8)
    max_len: int = field(
        default=512,
        metadata={
            "help":'输入数据编码后的最大长度，超过该长度会截取'
        }
    )

