# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved\n

import json
import logging
import os
from typing import Callable, Dict, Optional, Union

import torch
from huggingface_hub import ModelHubMixin, snapshot_download

logger = logging.getLogger(__name__)


class BaseModel(torch.nn.Module, ModelHubMixin):
    config_cls: Callable

    def device(self):
        return next(self.parameters()).device

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        cache_dir: str,
        force_download: bool,
        proxies: Optional[Dict] = None,
        resume_download: bool = False,
        local_files_only: bool,
        token: Union[str, bool, None],
        map_location: str = "cpu",
        strict: bool = True,
        revision: Optional[str] = None,
        **model_kwargs,
    ):
        if os.path.isdir(model_id):
            cached_model_dir = model_id
        else:
            cached_model_dir = snapshot_download(
                repo_id=model_id,
                revision=cls.revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                token=token,
                local_files_only=local_files_only,
            )

        with open(os.path.join(cached_model_dir, "config.json")) as fin:
            config = json.load(fin)

        for key, value in model_kwargs.items():
            if key in config:
                config[key] = value

        config = cls.config_cls(**config)
        print(f"[base] {cls.__name__}: building model architecture …", flush=True)
        model = cls(config)
        ckpt_path = os.path.join(cached_model_dir, "checkpoint.pt")
        print(f"[base] {cls.__name__}: loading checkpoint from {ckpt_path} …", flush=True)
        state_dict = torch.load(
            ckpt_path,
            weights_only=True,
            map_location=map_location,
        )
        print(f"[base] {cls.__name__}: applying state_dict …", flush=True)
        model.load_state_dict(state_dict, strict=strict)
        print(f"[base] {cls.__name__}: done.", flush=True)
        return model
