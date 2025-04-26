import transformers
import lm_eval
import typing
import torch
import utils


class LLaDA(lm_eval.api.model.LM):

    def __init__(
        self, 
        model_path   : str                  = "GSAI-ML/LLaDA-8B-Base",
        max_src_toks : typing.Optional[int] = None ,
        max_tgt_toks : int                  = 32   ,
        batch_size   : int                  = 4    ,
        seed         : int                  = 42   ,
    ):
        utils.set_seed(seed)
        super().__init__()

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code = True           ,
            torch_dtype       = torch.bfloat16 ,
            device_map        = "auto"         ,
        )

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        self.device       = self.model.hf_device_map[next(iter(self.model.hf_device_map.keys()))]
        self.mask_id      = 126336
        self.pad_id       = self.tokenizer.pad_token_id
        self.stop_id      = self.tokenizer.eos_token_id
        self.max_src_toks = max_src_toks
        self.max_tgt_toks = max_tgt_toks
        self.batch_size   = batch_size
        self.model.eval()
