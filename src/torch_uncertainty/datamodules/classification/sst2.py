import os

import torch

# TODO: Hm, why is it necessary?
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from datasets import DownloadConfig, load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from torch_uncertainty.datamodules.abstract import TUDataModule


# TODO: This should be put into utils
class HFTupleDataset(Dataset):
    def __init__(self, hf_ds, name=None):
        """Initialize the dataset wrapper for HuggingFace datasets."""
        self.ds = hf_ds
        self.dataset_name = (name or "dataset").lower().replace(" ", "_")

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.ds)

    def __getitem__(self, idx):
        """Get an item from the dataset."""
        item = self.ds[idx]
        x = {"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]}
        y = item.get("label", 0)
        return x, torch.tensor(int(y), dtype=torch.long)


class SST2DataModule(TUDataModule):
    num_classes = 2
    training_task = "classification"
    num_channels = 1
    input_shape = None  # TODO: fix

    def __init__(
        self,
        model_name="bert-base-uncased",
        max_len=128,
        batch_size=32,
        eval_batch_size=None,
        num_tta=1,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        local_files_only=False,
        eval_ood: bool = True,
    ):
        """Initialize the SST-2 data module."""
        super().__init__(
            root=".",
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            val_split=None,
            num_tta=num_tta,
            postprocess_set="val",
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        self.model_name = model_name
        self.max_len = max_len
        self.local_files_only = local_files_only
        self.eval_ood = eval_ood

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, local_files_only=local_files_only
        )

        self.near_oods, self.far_oods = [], []
        self.test_ood, self.val_ood = None, None

    def _wrap_tok_set(self, tokds, name):
        keep = [c for c in ["input_ids", "attention_mask", "label"] if c in tokds.column_names]
        tokds.set_format(type="torch", columns=keep)
        return HFTupleDataset(tokds, name=name)

    def _tok_single(self, raw, field, name):
        def tok(b):
            return self.tokenizer(
                b[field], max_length=self.max_len, truncation=True, padding="max_length"
            )

        cols_rm = [c for c in raw.column_names if c not in ("label", field)]
        tokds = raw.map(tok, batched=True, remove_columns=cols_rm)
        return self._wrap_tok_set(tokds, name)

    def _tok_pair(self, raw, field1, field2, name):
        def tok(b):
            return self.tokenizer(
                b[field1], b[field2], max_length=self.max_len, truncation=True, padding="max_length"
            )

        cols_rm = [c for c in raw.column_names if c not in ("label", field1, field2)]
        tokds = raw.map(tok, batched=True, remove_columns=cols_rm)
        return self._wrap_tok_set(tokds, name)

    def _tok_wmt16_en(self, raw, name):
        def tok(b):
            en = [t.get("en", "") for t in b["translation"]]
            return self.tokenizer(
                en, max_length=self.max_len, truncation=True, padding="max_length"
            )

        cols_rm = [c for c in raw.column_names if c not in ("label", "translation")]
        tokds = raw.map(tok, batched=True, remove_columns=cols_rm)
        return self._wrap_tok_set(tokds, name)

    def prepare_data(self):
        dl = DownloadConfig(local_files_only=self.local_files_only)
        # Cache SST-2
        load_dataset("glue", "sst2", download_config=dl)

        if not self.eval_ood:
            return

        load_dataset("yelp_polarity", split="test", download_config=dl)
        load_dataset("amazon_polarity", split="test", download_config=dl)

        load_dataset("ag_news", split="test", download_config=dl)
        load_dataset("SetFit/20_newsgroups", split="test", download_config=dl)
        load_dataset("SetFit/TREC-QC", split="test", download_config=dl)
        load_dataset("glue", "mnli", split="validation_mismatched", download_config=dl)
        load_dataset("glue", "rte", split="validation", download_config=dl)
        load_dataset("wmt16", "ro-en", split="test", download_config=dl)

    def setup(self, stage=None):
        dl = DownloadConfig(local_files_only=self.local_files_only)

        ds = load_dataset("glue", "sst2", download_config=dl)

        def tok_id(b):
            return self.tokenizer(
                b["sentence"], max_length=self.max_len, truncation=True, padding="max_length"
            )

        cols_rm = [c for c in ["sentence", "idx"] if c in ds["train"].column_names]
        tokds = ds.map(tok_id, batched=True, remove_columns=cols_rm)
        tokds.set_format(
            type="torch",
            columns=[
                c
                for c in ["input_ids", "attention_mask", "label"]
                if c in tokds["train"].column_names
            ],
        )

        if stage in (None, "fit"):
            self.train = HFTupleDataset(tokds["train"], name="sst2_train")
            self.val = HFTupleDataset(tokds["validation"], name="sst2_val")

        if stage in (None, "test"):
            self.test = HFTupleDataset(tokds["validation"], name="sst2_test")
            self.near_oods, self.far_oods = [], []

            if self.eval_ood:
                # -------- Near OOD --------
                yelp_raw = load_dataset("yelp_polarity", split="test", download_config=dl)
                amazon_raw = load_dataset("amazon_polarity", split="test", download_config=dl)

                self.near_oods.append(self._tok_single(yelp_raw, "text", "yelp_polarity"))
                self.near_oods.append(self._tok_single(amazon_raw, "content", "amazon_polarity"))

                # -------- Far OOD --------
                ag_raw = load_dataset("ag_news", split="test", download_config=dl)
                n20_raw = load_dataset("SetFit/20_newsgroups", split="test", download_config=dl)
                trec_raw = load_dataset("SetFit/TREC-QC", split="test", download_config=dl)
                mnli_raw = load_dataset(
                    "glue", "mnli", split="validation_mismatched", download_config=dl
                )
                rte_raw = load_dataset("glue", "rte", split="validation", download_config=dl)
                wmt_raw = load_dataset("wmt16", "ro-en", split="test", download_config=dl)

                self.far_oods.append(self._tok_single(ag_raw, "text", "ag_news"))
                self.far_oods.append(self._tok_single(n20_raw, "text", "20_newsgroups"))
                self.far_oods.append(self._tok_single(trec_raw, "text", "trec_qc"))
                self.far_oods.append(self._tok_pair(mnli_raw, "premise", "hypothesis", "mnli_mm"))
                self.far_oods.append(self._tok_pair(rte_raw, "sentence1", "sentence2", "rte"))
                self.far_oods.append(self._tok_wmt16_en(wmt_raw, "wmt16_ro_en_en"))

                self.test_ood = self.test
                self.val_ood = None

    def train_dataloader(self):
        return self._data_loader(self.train, training=True, shuffle=True)

    def val_dataloader(self):
        return self._data_loader(self.val, training=False)

    def test_dataloader(self):
        loaders = [self._data_loader(self.test, training=False)]
        if self.eval_ood:
            loaders.append(self._data_loader(self.test_ood, training=False))
            # no val_ood loader
            loaders.extend(self._data_loader(ds, training=False) for ds in self.near_oods)
            loaders.extend(self._data_loader(ds, training=False) for ds in self.far_oods)
        return loaders

    def get_test_set(self):
        return self.test

    def get_indices(self):
        # TODO: Needs comments, at least
        idx = 0
        out = {"test": [idx]}
        idx += 1
        if self.eval_ood:
            out["test_ood"] = [idx]
            idx += 1
            out["val_ood"] = []  # kept empty
            out["near_oods"] = list(range(idx, idx + len(self.near_oods)))
            idx += len(self.near_oods)
            out["far_oods"] = list(range(idx, idx + len(self.far_oods)))
            idx += len(self.far_oods)
        else:
            out |= {"test_ood": [], "val_ood": [], "near_oods": [], "far_oods": []}
        out["shift"] = []
        return out
