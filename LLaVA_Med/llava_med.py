import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava import LlavaLlamaForCausalLM
from llava.conversation import conv_templates
from llava.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria

from PIL import Image
import random
import math
import pandas as pd
import yaml

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

def patch_config(config):
    patch_dict = {
        "use_mm_proj": True,
        "mm_vision_tower": "openai/clip-vit-large-patch14",
        "mm_hidden_size": 1024
    }

    cfg = AutoConfig.from_pretrained(config)
    if not hasattr(cfg, "mm_vision_tower"):
        print(f'`mm_vision_tower` not found in `{config}`, applying patch and save to disk.')
        for k, v in patch_dict.items():
            setattr(cfg, k, v)
        cfg.save_pretrained(config)

def get_input_prompt(args):
    # Read prompt_dict yaml file
    print("Reading prompt_dict yaml file")
    with open("prompt_dict.yaml") as file:
        yaml_data = yaml.safe_load(file)

    print("CHOSEN PROMPT: ", yaml_data[args.prompt_type])
    return yaml_data[args.prompt_type]

class LlavaMedInference:
    def __init__(self, model_path="/data/mn27889/.cache/huggingface/hub/llava_med_pvqa" ):
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        disable_torch_init()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = LlavaLlamaForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.float16, use_cache=True).cuda()
        self.image_processor = CLIPImageProcessor.from_pretrained(self.model.config.mm_vision_tower, torch_dtype=torch.float16)
        vision_tower = self.model.model.vision_tower[0]
        vision_tower.to(device='cuda', dtype=torch.float16)
        
        # read from config
        mm_use_im_start_end = getattr(self.model.config, "mm_use_im_start_end", False)
        self.tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            self.tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

        # import pdb; pdb.set_trace()
        vision_config = vision_tower.config
        vision_config.im_patch_token = self.tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        vision_config.use_im_start_end = mm_use_im_start_end
        if mm_use_im_start_end:
            vision_config.im_start_token, vision_config.im_end_token = self.tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
        self.image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2
        
    def forward(self, image: torch.Tensor, qs: str) -> str:
        

        # image preprocessing
        # image = Image.open(image_file) 
        image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        images = image_tensor.unsqueeze(0).half().cuda()

        # prompt preprocessing
        if getattr(self.model.config, 'mm_use_im_start_end', False):
            qs = qs + '\n' + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * self.image_token_len + DEFAULT_IM_END_TOKEN
        else:
            qs = qs + '\n' + DEFAULT_IMAGE_PATCH_TOKEN * self.image_token_len

        conv = conv_templates["simple"].copy()
        conv.append_message(conv.roles[0], qs)
        prompt = conv.get_prompt()
        inputs = self.tokenizer([prompt])

        input_ids = torch.as_tensor(inputs.input_ids).cuda()

        keywords = ['###']
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)


        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images,
                do_sample=True,
                temperature=0.7,
                max_new_tokens=1024,
                stopping_criteria=[stopping_criteria])

        # TODO: new implementation
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] Sample {i}: {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]

        try:
            index = outputs.index(conv.sep)
        except ValueError:
            outputs += conv.sep
            index = outputs.index(conv.sep)

        outputs = outputs[:index].strip()
        return outputs

if __name__ == "__main__":
    inference = LlavaMedInference()

    # Read the image
    image_path = "/data/mn27889/pvqa/images/test/test_0149.jpg"  # Change this to the path of your image
    image = Image.open(image_path)

    # Assuming you have the question string
    question = "Describe the pathology in detail."

    # Run inference
    result = inference.forward(image, question)
    print(result)

