import models
import lm_eval
import pprint


results = lm_eval.simple_evaluate(
    model = models.AutoregressiveLLaDA(
        max_src_toks = None,
        max_tgt_toks = 256 ,
        batch_size   = 4   ,
    ),
    tasks=["gsm8k"], #, "gpqa_main_zeroshot"],
    limit = 100, # TODO: remove this
    num_fewshot = 1,

)

pprint.pp(results)
pprint.pprint(results["results"])

# save result on file
with open("results.log", "w") as f:
    pprint.pprint(results, stream=f)
    pprint.pprint(results["results"], stream=f)
