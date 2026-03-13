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
        logger.info("%s: building model architecture …", cls.__name__)
        model = cls(config)
        ckpt_path = os.path.join(cached_model_dir, "checkpoint.pt")
        logger.info("%s: loading checkpoint from %s …", cls.__name__, ckpt_path)
        state_dict = torch.load(
            ckpt_path,
            weights_only=True,
            map_location=map_location,
        )
        logger.info("%s: applying state_dict …", cls.__name__)
        model.load_state_dict(state_dict, strict=strict)
        logger.info("%s: done.", cls.__name__)
        return model
