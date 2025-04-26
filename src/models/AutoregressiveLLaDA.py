import models
import torch
import tqdm

class AutoregressiveLLaDA(models.LLaDA):
    """
    Simple autoregressive decoding strategy for LLaDA.
    """

    def __init__(self, *args, **kwargs):
        super(AutoregressiveLLaDA, self).__init__(*args, **kwargs)


    @torch.no_grad()
    def loglikelihood(self, requests):
        raise NotImplementedError()

    @torch.no_grad()
    def loglikelihood_rolling(self, requests):
        raise NotImplementedError()

    @torch.no_grad()
    def generate_until(self, requests):

        results  = []
        requests = [requests[i:i + self.batch_size] for i in range(0, len(requests), self.batch_size)]
        for batch in tqdm.tqdm(requests):

            # encode inputs
            batch = [request.args[0] for request in batch]
            batch = self.tokenizer(batch)["input_ids"]

            # generate answers
            answers = [[] for _ in range(len(batch))]
            for _ in range(self.max_tgt_toks):

                # generate token
                source  = [torch.tensor(src + [self.mask_id]) for src in batch]
                source  = torch.nn.utils.rnn.pad_sequence(source, batch_first=True, padding_value=self.pad_id)
                logits  = self.model(source).logits[:,-1,:]
                preds   = logits.argmax(-1)
                batch   = [src + [prd] for src,prd in zip(batch,preds)]
                answers = [ans + [prd] for ans,prd in zip(answers,preds)]

            # decode and append to results
            answers = self.tokenizer.decode(answers)
            answers.extend(answers)

        return results
