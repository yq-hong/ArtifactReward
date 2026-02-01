import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "t2v_metrics")))
import torch
from tqdm import tqdm
from t2v_metrics.models.vqascore_models.mm_utils import expand2square, load_pretrained_model, t5_tokenizer_image_token
from t2v_metrics.models.vqascore_models.clip_t5.model import CLIPT5ForConditionalGeneration, ModelArguments

CONTEXT_LEN = 2048
SYSTEM_MSG = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
IGNORE_INDEX = -100
# IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
MAX_BATCH = 8


class VQAScore:
    def __init__(self, args):
        self.ckpt_path = args.vqa_ckpt_path
        if 'flan-t5-xl' in self.ckpt_path:
            self.model_name = 'clip-flant5-xl'
            self.tokenizer_path = 'google/flan-t5-xl'
        else:
            self.model_name = 'clip-flant5-xxl'
            self.tokenizer_path = 'google/flan-t5-xxl'

        CLIP_T5_MODELS = {
            # We recommend using 'clip-flant5-xxl' for maximal performance.
            # If you want to use a smaller model, we recommend using 'clip-flant5-xl'.
            'clip-flant5-xxl': {'tokenizer': {'path': 'google/flan-t5-xxl',
                                              'model_max_length': CONTEXT_LEN},
                                'model': {'path': self.ckpt_path,
                                          'conversation': 't5_chat',
                                          'image_aspect_ratio': 'pad'}, },
            'clip-flant5-xl': {'tokenizer': {'path': 'google/flan-t5-xl',
                                             'model_max_length': CONTEXT_LEN},
                               'model': {'path': self.ckpt_path,
                                         'conversation': 't5_chat',
                                         'image_aspect_ratio': 'pad'}}
        }
        self.model_max_length = CONTEXT_LEN
        self.image_aspect_ratio = 'pad'
        self.conversational_style = 't5_chat'
        self.context_len = CONTEXT_LEN
        self.model_args = ModelArguments()

        self.question_template = 'Does this figure show "{}"? Please answer yes or no.'
        self.answer_template = "Yes"

    @property
    def __name__(self):
        return 'VQAScore'

    def format_question(self, question, conversation_style='plain'):
        if conversation_style == 't5_plain':  # for 1st stage t5 model
            question = DEFAULT_IMAGE_TOKEN + question
        elif conversation_style == 't5_chat':  # for 2nd stage t5 model
            question = SYSTEM_MSG + " USER: " + DEFAULT_IMAGE_TOKEN + "\n" + question + " ASSISTANT: "
        elif conversation_style == 't5_chat_no_system':  # for 2nd stage t5 model
            question = "USER: " + DEFAULT_IMAGE_TOKEN + "\n" + question + " ASSISTANT: "
        elif conversation_style == 't5_chat_no_system_no_user':  # for 2nd stage t5 model
            question = "" + DEFAULT_IMAGE_TOKEN + "\n" + question + " : "
        else:
            raise NotImplementedError()
        return question

    def load_to_device(self, load_device):
        self.device = load_device
        self.tokenizer, self.model, self.image_processor = load_pretrained_model(CLIPT5ForConditionalGeneration,
                                                                                 self.model_args,
                                                                                 model_path=self.ckpt_path,
                                                                                 tokenizer_path=self.tokenizer_path,
                                                                                 model_max_length=self.model_max_length,
                                                                                 padding_side=None,
                                                                                 image_aspect_ratio=self.image_aspect_ratio,
                                                                                 mmprojector_repo=None,
                                                                                 mmprojector_name=None,
                                                                                 device=load_device)

        self.model = self.model.to(load_device)
        self.model.eval()

    def __call__(self, prompts, images, **kwargs):
        # image_list is a list of PIL image
        # Adapted from https://github.com/linzhiqiu/t2v_metrics/blob/main/t2v_metrics/models/vqascore_models/clip_t5_model.py
        self.device = list(self.model.parameters())[0].device
        result = []
        for prompt, image in tqdm(zip(prompts, images), total=len(prompts)):
            with torch.no_grad():
                question = self.question_template.format(prompt)
                answer = self.answer_template.format(prompt)
                question = self.format_question(question, conversation_style=self.conversational_style)
                input_id = [t5_tokenizer_image_token(question, self.tokenizer, return_tensors='pt')]
                label = [t5_tokenizer_image_token(answer, self.tokenizer, return_tensors='pt')]
                input_id = torch.nn.utils.rnn.pad_sequence(input_id,
                                                           batch_first=True,
                                                           padding_value=self.tokenizer.pad_token_id)
                label = torch.nn.utils.rnn.pad_sequence(label,
                                                        batch_first=True,
                                                        padding_value=IGNORE_INDEX)
                input_id = input_id[:, :self.tokenizer.model_max_length]
                label = label[:, :self.tokenizer.model_max_length]  # [1, token_size]
                attention_mask = input_id.ne(self.tokenizer.pad_token_id)
                decoder_attention_mask = label.ne(IGNORE_INDEX)

                input_id, attention_mask, decoder_attention_mask, label = input_id.to(self.device), \
                    attention_mask.to(self.device), decoder_attention_mask.to(self.device), label.to(self.device)

                if self.image_aspect_ratio == 'pad':
                    image = expand2square(image, tuple(int(x * 255) for x in self.image_processor.image_mean))
                image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values']
                image = image.to(self.device, dtype=self.model.dtype)

                model_input_kwargs = {'input_ids': input_id,
                                      'attention_mask': attention_mask,
                                      'decoder_attention_mask': decoder_attention_mask,
                                      'labels': label,
                                      'images': image,
                                      'past_key_values': None,
                                      'inputs_embeds': None,
                                      'use_cache': None,
                                      'output_attentions': None,
                                      'output_hidden_states': None,
                                      'return_dict': True}
                outputs = self.model(**model_input_kwargs)
                logits = outputs.logits  # [1, token_size, vocab_size]

                loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
                logits_flat = logits.view(-1, logits.size(-1))  # [batch*seq_len, vocab_size]
                labels_flat = label.view(-1)  # [batch*seq_len]
                lm_prob = (-loss_fct(logits_flat, labels_flat)).exp()

            result.append(lm_prob.item())
        return result


class VQAScoreBatch:
    def __init__(self, args):
        self.ckpt_path = args.vqa_ckpt_path
        if 'flan-t5-xl' in self.ckpt_path:
            self.model_name = 'clip-flant5-xl'
            self.tokenizer_path = 'google/flan-t5-xl'
        else:
            self.model_name = 'clip-flant5-xxl'
            self.tokenizer_path = 'google/flan-t5-xxl'

        self.model_max_length = CONTEXT_LEN
        self.image_aspect_ratio = 'pad'
        self.conversational_style = 't5_chat'
        self.context_len = CONTEXT_LEN
        self.model_args = ModelArguments()

        self.question_template = 'Does this figure show "{}"? Please answer yes or no.'
        self.answer_template = "Yes"

    @property
    def __name__(self):
        return 'VQAScoreBatch'

    def format_question(self, question, conversation_style='plain'):
        if conversation_style == 't5_plain':  # for 1st stage t5 model
            question = DEFAULT_IMAGE_TOKEN + question
        elif conversation_style == 't5_chat':  # for 2nd stage t5 model
            question = SYSTEM_MSG + " USER: " + DEFAULT_IMAGE_TOKEN + "\n" + question + " ASSISTANT: "
        elif conversation_style == 't5_chat_no_system':  # for 2nd stage t5 model
            question = "USER: " + DEFAULT_IMAGE_TOKEN + "\n" + question + " ASSISTANT: "
        elif conversation_style == 't5_chat_no_system_no_user':  # for 2nd stage t5 model
            question = "" + DEFAULT_IMAGE_TOKEN + "\n" + question + " : "
        else:
            raise NotImplementedError()
        return question

    def load_to_device(self, load_device):
        self.device = load_device
        self.tokenizer, self.model, self.image_processor = load_pretrained_model(CLIPT5ForConditionalGeneration,
                                                                                 self.model_args,
                                                                                 model_path=self.ckpt_path,
                                                                                 tokenizer_path=self.tokenizer_path,
                                                                                 model_max_length=self.model_max_length,
                                                                                 padding_side=None,
                                                                                 image_aspect_ratio=self.image_aspect_ratio,
                                                                                 mmprojector_repo=None,
                                                                                 mmprojector_name=None,
                                                                                 device=load_device)

        self.model = self.model.to(load_device)
        self.model.eval()

    def batch_score(self, prompts, images):
        # image_list is a list of PIL image
        # Adapted from https://github.com/linzhiqiu/t2v_metrics/blob/main/t2v_metrics/models/vqascore_models/clip_t5_model.py
        self.device = list(self.model.parameters())[0].device

        with torch.no_grad():
            questions = [self.question_template.format(p) for p in prompts]
            answers = [self.answer_template.format(p) for p in prompts]
            questions = [self.format_question(q, conversation_style=self.conversational_style) for q in questions]

            input_ids = [t5_tokenizer_image_token(q, self.tokenizer, return_tensors='pt') for q in questions]
            labels = [t5_tokenizer_image_token(a, self.tokenizer, return_tensors='pt') for a in answers]
            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids,
                                                        batch_first=True,
                                                        padding_value=self.tokenizer.pad_token_id)
            labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                     batch_first=True,
                                                     padding_value=IGNORE_INDEX)
            input_ids = input_ids[:, :self.tokenizer.model_max_length]
            labels = labels[:, :self.tokenizer.model_max_length]  # [batch, token_size]

            attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
            decoder_attention_mask = labels.ne(IGNORE_INDEX)

            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            attention_mask = attention_mask.to(self.device)
            decoder_attention_mask = decoder_attention_mask.to(self.device)

            if self.image_aspect_ratio == 'pad':
                images = [expand2square(img, tuple(int(x * 255) for x in self.image_processor.image_mean)) for img in
                          images]
            images = [self.image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0] for img in images]
            images = torch.stack(images, dim=0)
            images = images.to(self.device, dtype=self.model.dtype)

            model_input_kwargs = {'input_ids': input_ids,
                                  'attention_mask': attention_mask,
                                  'decoder_attention_mask': decoder_attention_mask,
                                  'labels': labels,
                                  'images': images,
                                  'past_key_values': None,
                                  'inputs_embeds': None,
                                  'use_cache': None,
                                  'output_attentions': None,
                                  'output_hidden_states': None,
                                  'return_dict': True}
            outputs = self.model(**model_input_kwargs)
            logits = outputs.logits  # [batch, token_size, vocab_size]

            loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
            lm_prob = torch.zeros(logits.shape[0])
            for k in range(lm_prob.shape[0]):
                lm_prob[k] = (-loss_fct(logits[k], labels[k])).exp()

        return lm_prob

    def __call__(self, prompts, images, **kwargs):
        all_lm_probs = []

        for i in tqdm(range(0, len(prompts), MAX_BATCH)):
            micro_prompts = prompts[i:i + MAX_BATCH]
            micro_images = images[i:i + MAX_BATCH]

            lm_prob = self.batch_score(micro_prompts, micro_images)
            all_lm_probs.append(lm_prob)

        lm_probs = torch.cat(all_lm_probs, dim=0)

        return lm_probs.tolist()
