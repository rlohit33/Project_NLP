# -*- coding: utf-8 -*-
"""T5_questionGenerator.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1D6pip4l3VZDJkdnF0GzCEb3Frl-aQj5M
"""

import itertools
import logging
from typing import Optional, Dict, Union
import nltk
nltk.download('punkt')
import re

from nltk import sent_tokenize

import torch
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer,PreTrainedModel,PreTrainedTokenizer

logger = logging.getLogger(__name__)

class QGPipeline:
    
    def __init__(
        self,
        model_choosen: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        ans_model: PreTrainedModel,
        ans_tokenizer: PreTrainedTokenizer,
        qg_format: str,
        use_cuda: bool
    ):
        
        self.model_choosen = model_choosen
        self.tokenizer = tokenizer

        self.ans_model = ans_model
        self.ans_tokenizer = ans_tokenizer

        self.qg_format = qg_format

        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.model_choosen.to(self.device)
        
        if self.ans_model is not self.model_choosen:
            self.ans_model.to(self.device)

        assert self.model_choosen.__class__.__name__ in ["T5ForConditionalGeneration", "BartForConditionalGeneration"]
        
        if "T5ForConditionalGeneration" in self.model_choosen.__class__.__name__:
            self.model_type = "t5"
        else:
            self.model_type = "bart"

    def __call__(self, inputs: str):
        inputs = " ".join(inputs.split())
        sents, answers = self._extract_answers(inputs)
        flat_answers = list(itertools.chain(*answers))
        
        if len(flat_answers) == 0:
          return []

        if self.qg_format == "prepend":
            
            qg_examples = self._prepare_inputs_for_qg_from_answers_prepend(inputs, answers)
            
        else:
            qg_examples = self._prepare_inputs_for_qg_from_answers_hl(sents, answers)
        
        qg_inputs = [example['source_text'] for example in qg_examples]
        ans_index_start = [example['ans_start_idx'] for example in qg_examples]
        ans_index_end = [example['ans_end_idx'] for example in qg_examples]
        questions = self._generate_questions(qg_inputs)
        output = [{'answer': example['answer'], 'question': que,'start_index':index_strt,'end_index':index_end} for example, que, index_strt,index_end in zip(qg_examples, questions,ans_index_start,ans_index_end)]
        return output
    
    def _generate_questions(self, inputs):
        inputs = self._tokenize(inputs, padding=True, truncation=True)
        
        outs = self.model_choosen.generate(
            input_ids=inputs['input_ids'].to(self.device), 
            attention_mask=inputs['attention_mask'].to(self.device), 
            max_length=32,
            num_beams=4,
        )
        
        questions = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        return questions
    
    def _extract_answers(self, context):
        sents, inputs = self._prepare_inputs_for_ans_extraction(context)

        inputs = self._tokenize(inputs, padding=True, truncation=True)

        outs = self.ans_model.generate(
            input_ids=inputs['input_ids'].to(self.device), 
            attention_mask=inputs['attention_mask'].to(self.device), 
            max_length=32,
        )

        
        dec = [self.ans_tokenizer.decode(ids, skip_special_tokens=False) for ids in outs]
        
        new_dec = []
        for d in dec:
          d = re.sub("<pad> ","",d)
          d = re.sub("<pad>","",d)
          new_dec.append(d)
        

        answers = [item.split('<sep>') for item in new_dec]
        
        answers = [i[:-1] for i in answers]
        
        return sents, answers
    
    def _tokenize(self,
        inputs,
        padding=True,
        truncation=True,
        add_special_tokens=True,
        max_length=512
    ):
        inputs = self.tokenizer.batch_encode_plus(
            inputs, 
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            padding="max_length" if padding else False,
            pad_to_max_length=padding,
            return_tensors="pt"
        )
        return inputs
    
    def _prepare_inputs_for_ans_extraction(self, text):
        sents = sent_tokenize(text)
        
        inputs = []
        for i in range(len(sents)):
            source_text = "extract answers:"
            for j, sent in enumerate(sents):
                if i == j:
                    
                    sent = "<hl> %s <hl>" % sent
                    
                    source_text = "%s %s" % (source_text, sent)
                    source_text = source_text.strip()
            
            if self.model_type == "t5":
                
                source_text = source_text + " </s>"
            inputs.append(source_text)
            
        return sents, inputs
    
    def _prepare_inputs_for_qg_from_answers_hl(self, sents, answers):
        inputs = []

        for i, answer in enumerate(answers):
            
            if len(answer) == 0: continue
            
            for answer_text in answer:
                answer_text = re.sub("<pad> ","",answer_text)
                sent = sents[i]
                sents_copy = sents[:]
                
                answer_text = answer_text.strip()
                if answer_text in sent:
                   ans_start_idx = sent.index(answer_text) 
                else: 
                  continue

                #ans_start_idx = sent.index(answer_text)
                
                sent = f"{sent[:ans_start_idx]} <hl> {answer_text} <hl> {sent[ans_start_idx + len(answer_text): ]}"
                sents_copy[i] = sent
                
                source_text = " ".join(sents_copy)
                source_text = f"generate question: {source_text}" 
                if self.model_type == "t5":
                    source_text = source_text + " </s>"
                
                len_ans = len(answer_text.split(" "))
                inputs.append({"answer": answer_text, "source_text": source_text, "ans_start_idx":ans_start_idx, "ans_end_idx": (ans_start_idx +(len_ans - 1)) })
        
        return inputs
   
    def _prepare_inputs_for_qg_from_answers_prepend(self, context, answers):
        flat_answers = list(itertools.chain(*answers))
        examples = []
        for answer in flat_answers:
            source_text = f"answer: {answer} context: {context}"
            if self.model_type == "t5":
                source_text = source_text + " </s>"
            
            examples.append({"answer": answer, "source_text": source_text})
        return examples
            
    
      
Task_details = {
    "generate-question": {
        "implement": QGPipeline,
        "default": {
            "model_choosen": "valhalla/t5-small-qg-hl",
            "ans_model": "valhalla/t5-small-qa-qg-hl",
        }
    },
}

def get_question_pipeline(
    task: str,
    model_choosen: Optional = None,
    tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
    qg_format: Optional[str] = "highlight",
    ans_model: Optional = None,
    ans_tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
    use_cuda: Optional[bool] = True,
    **kwargs,
):
    # Retrieve the task
    if task not in Task_details:
        raise KeyError("Unknown task {}, available tasks are {}".format(task, list(Task_details.keys())))

    task_to_perform = Task_details[task]
    implementation_class = task_to_perform["implement"]

    # Use default model/config/tokenizer for the task if no model is provided
    if model_choosen is None:
        model_choosen = task_to_perform["default"]["model_choosen"]
    
    # Try to infer tokenizer from model or config name (if provided as str)
    if tokenizer is None:
        if isinstance(model_choosen, str):
            tokenizer = model_choosen
          
        else:
            # Impossible to guest what is the right tokenizer here
            raise Exception(
                "Impossible to guess which tokenizer to use. "
                "Please provided a PretrainedTokenizer class or a path/identifier to a pretrained tokenizer."
            )
    
    # Instantiate tokenizer if needed
    if isinstance(tokenizer, (str, tuple)):
        if isinstance(tokenizer, tuple):
            # For tuple we have (tokenizer name, {kwargs})
            tokenizer = AutoTokenizer.from_pretrained(tokenizer[0], **tokenizer[1],use_fast=True)

        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            
    
    # Instantiate model if needed
    if isinstance(model_choosen, str):
        model_choosen = AutoModelForSeq2SeqLM.from_pretrained(model_choosen)
    
    if task == "generate-question":
        if ans_model is None:
            # load default ans model
            ans_model = task_to_perform["default"]["ans_model"]
            ans_tokenizer = AutoTokenizer.from_pretrained(ans_model,use_fast=True)
            ans_model = AutoModelForSeq2SeqLM.from_pretrained(ans_model)
        else:
            # Try to infer tokenizer from model or config name (if provided as str)
            if ans_tokenizer is None:
                if isinstance(ans_model, str):
                    ans_tokenizer = ans_model
                else:
                    # Impossible to guess what is the right tokenizer here
                    raise Exception(
                        "Impossible to guess which tokenizer to use. "
                        "Please provided a PretrainedTokenizer class or a path/identifier to a pretrained tokenizer."
                    )
            
            # Instantiate tokenizer if needed
            if isinstance(ans_tokenizer, (str, tuple)):
                if isinstance(ans_tokenizer, tuple):
                    # For tuple we have (tokenizer name, {kwargs})
                    ans_tokenizer = AutoTokenizer.from_pretrained(ans_tokenizer[0], **ans_tokenizer[1],use_fast=True)
                else:
                    ans_tokenizer = AutoTokenizer.from_pretrained(ans_tokenizer,use_fast=True)

            if isinstance(ans_model, str):
                ans_model = AutoModelForSeq2SeqLM.from_pretrained(ans_model)

    if task == "generate-question":
        return implementation_class(model_choosen=model_choosen, tokenizer=tokenizer, ans_model=ans_model, ans_tokenizer=ans_tokenizer, qg_format=qg_format, use_cuda=use_cuda)
    else:
        return implementation_class(model_choosen=model_choosen, tokenizer=tokenizer, ans_model=model_choosen, ans_tokenizer=tokenizer, qg_format=qg_format, use_cuda=use_cuda)

#Questions = get_question_pipeline("generate-question")
#answer_question = Questions( 'HIS "Hightech Information System Limited" established 1987 , is a Hong Kong based graphics card manufacturer that produces AMD formerly known as ATI Radeon graphics cards. \
 Its headquarters are in Hong Kong, with additional sales offices and distribution networks in Europe, the Middle East, North America and Asia Pacific Regions.\
 The current distributor in Hong Kong is JunMax Technology. Products HIS manufactures and sells AMD Radeon series video cards. \
 They are known for their IceQ cooling technology as well as producing the latest and fastest PCI cards like AMD Radeon RX 590, RX 5700 and RX 5700 XT. \
 In 2019, HIS launched new versions of the RX 5700 XT in pink and blue. External links HIS Ltd.')
#answer_question