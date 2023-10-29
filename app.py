from typing import List, Optional

from flask import Flask, request
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_id_or_path = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)
model = AutoModelForCausalLM.from_pretrained(model_id_or_path)
model.to(device)

SECRET_KEY = 'your-secret-key'

EOT_TOKEN_ID = model.config.eos_token_id
MAX_LENGTH = model.config.max_length
MAX_GEN_TOKS = 128  # same as max_new_tokens

app = Flask(__name__)


class MultiTokenEOSCriteria(transformers.StoppingCriteria):
    """Criteria to stop on the specified multi-token sequence."""

    def __init__(
        self,
        sequence: str,
        tokenizer: transformers.PreTrainedTokenizer,
        initial_decoder_input_length: int,
        batch_size: int,
    ):
        self.initial_decoder_input_length = initial_decoder_input_length
        self.done_tracker = [False] * batch_size
        self.sequence = sequence
        self.sequence_ids = tokenizer.encode(
            sequence, add_special_tokens=False)
        self.sequence_id_len = len(self.sequence_ids)
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # For efficiency, we compare the last n tokens where n is the number of tokens in the stop_sequence
        lookback_ids_batch = input_ids[:, self.initial_decoder_input_length:][
            :, -self.sequence_id_len:
        ]

        lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)

        for i, done in enumerate(self.done_tracker):
            if not done:
                self.done_tracker[i] = self.sequence in lookback_tokens_batch[i]
        return False not in self.done_tracker


def stop_sequences_criteria(
    tokenizer: transformers.PreTrainedTokenizer,
    stop_sequences: List[str],
    initial_decoder_input_length: int,
    batch_size: int,
) -> transformers.StoppingCriteriaList:
    return transformers.StoppingCriteriaList(
        [
            *[
                MultiTokenEOSCriteria(
                    sequence, tokenizer, initial_decoder_input_length, batch_size
                )
                for sequence in stop_sequences
            ],
        ]
    )


@app.route('/config', methods=['GET'])
def get_config():
    return {
        'eot_token_id': EOT_TOKEN_ID,
        'max_length': MAX_LENGTH,
        'max_gen_toks': MAX_GEN_TOKS
    }


@app.route('/tok_encode', methods=['POST'])
def tok_encode():
    data = request.get_json()
    text = data['text']
    return tokenizer.encode(text)


@app.route('/tok_decode', methods=['POST'])
def tok_decode():
    data = request.get_json()
    ids = data['ids']
    return tokenizer.decode(ids)


@app.route('/greedy_util', methods=['POST'])
def greedy_util():
    data = request.get_json()
    context: str = data['context']
    stop: Optional[List[str]] = data.get('stop', None)

    input_ids = tokenizer(context, return_tensors='pt')['input_ids'].to(device)

    stopping_criteria = stop_sequences_criteria(
        tokenizer, stop, input_ids.shape[1], input_ids.shape[0]
    )

    generations = model.generate(
        input_ids=input_ids,
        temperature=0.0,
        max_new_tokens=MAX_GEN_TOKS,
        stopping_criteria=stopping_criteria,
        do_sample=False,
    )

    generations = generations.tolist()[0][len(input_ids[0]):]

    response = tokenizer.decode(generations)

    return response


@app.route('/loglikelihood_tokens', methods=['POST'])
def loglikelihood_tokens():
    data = request.get_json()
    context_enc: List[int] = data['context_enc']
    continuation_enc: List[int] = data['continuation_enc']

    inp = (context_enc + continuation_enc)[-(MAX_LENGTH + 1) :]
    ctxlen = len(context_enc) - max(0, len(context_enc) + len(continuation_enc) - (MAX_LENGTH + 1))

    input_ids = torch.tensor(inp).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_ids.to(device), labels=input_ids)
        logits = outputs.logits.cpu().detach()

    logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
    token_logprobs = torch.gather(logprobs, 2, input_ids.unsqueeze(-1)).squeeze(-1)
    continuation_logprobs = torch.sum(token_logprobs[:, ctxlen:])

    is_greedy = (logits.argmax(-1)[0, ctxlen-1:-1] == input_ids[0, ctxlen:]).all()

    return {
        "continuation_logprobs": continuation_logprobs.item(),
        "is_greedy": is_greedy.item(),
    }

@app.before_request
def check_secret_key():
    if request.endpoint != 'get_config':
        if request.headers.get('X-Secret-Key') != SECRET_KEY:
            return 'Invalid secret key', 403

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
