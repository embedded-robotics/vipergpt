# From: https://github.com/microsoft/LLaVA-Med/blob/main/llava/eval/model_vqa_med.py

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

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1" # Suppress HF warnings

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

def eval_model(args):
    # Model
    disable_torch_init()
    model_name = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if args.mm_projector is None:
        patch_config(model_name)
        
        print(model_name)

        if "BiomedCLIP" in model_name or "biomed_clip" in model_name:
            model = LlavaLlamaForCausalLM.from_pretrained(model_name, use_cache=True).cuda()
            model = model.to(torch.float16)
            image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
            
            openai_vision_tower = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
            vision_config = openai_vision_tower.config
            vision_tower = model.model.vision_tower[0]
            vision_tower.to(device='cuda', dtype=torch.float16)
            setattr(vision_tower, 'config', vision_config)
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, use_cache=True).cuda()
            image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=torch.float16)
            vision_tower = model.model.vision_tower[0]
            vision_tower.to(device='cuda', dtype=torch.float16)
    
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

        # import pdb; pdb.set_trace()
        vision_config = vision_tower.config
        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        vision_config.use_im_start_end = mm_use_im_start_end
        if mm_use_im_start_end:
            vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
        image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2
    
    else:
        # in case of using a pretrained model with only a MLP projector weights
        model = LlavaLlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, use_cache=True).cuda()

        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

        vision_tower = CLIPVisionModel.from_pretrained(args.vision_tower, torch_dtype=torch.float16).cuda()

        if "BiomedCLIP" in model.config.mm_vision_tower:
            image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
        else:
            image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=torch.float16)

        vision_config = vision_tower.config
        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        vision_config.use_im_start_end = mm_use_im_start_end
        if mm_use_im_start_end:
            vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

        image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2

        mm_projector = torch.nn.Linear(vision_config.hidden_size, model.config.hidden_size)
        mm_projector_weights = torch.load(args.mm_projector, map_location='cpu')
        mm_projector.load_state_dict({k.split('.')[-1]: v for k, v in mm_projector_weights.items()})

        model.model.mm_projector = mm_projector.cuda().half()
        model.model.vision_tower = [vision_tower]

    qs = "Describe the following image in detail. If there is a pathology, describe its location in detail. If not, describe the normal anatomy in detail."
    # qs = get_input_prompt(args)
    qs = qs.replace('<image>', '').strip()
    cur_prompt = qs

    # LOADING A SAMPLE IMAGE
    
    image_file = args.img_path
    image = Image.open(image_file)
    # print(image.size)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    images = image_tensor.unsqueeze(0).half().cuda()

    if getattr(model.config, 'mm_use_im_start_end', False):
        qs = qs + '\n' + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN
    else:
        qs = qs + '\n' + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
    cur_prompt = cur_prompt + '\n' + '<image>'
    # print("CUR PROMPT: ", cur_prompt)

    if args.conv_mode == 'simple_legacy':
        qs += '\n\n### Response:'

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    prompt = conv.get_prompt()
    inputs = tokenizer([prompt])

    input_ids = torch.as_tensor(inputs.input_ids).cuda()

    keywords = ['###']
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
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
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    # print("\n")
    # print(outputs)

    if args.conv_mode == 'simple_legacy':
        while True:
            cur_len = len(outputs)
            outputs = outputs.strip()
            for pattern in ['###', 'Assistant:', 'Response:']:
                if outputs.startswith(pattern):
                    outputs = outputs[len(pattern):].strip()
            if len(outputs) == cur_len:
                break

    try:
        index = outputs.index(conv.sep)
    except ValueError:
        outputs += conv.sep
        index = outputs.index(conv.sep)

    outputs = outputs[:index].strip()
    print(outputs)

    # prompt for answer
    print('mimi test ', args.answer_prompter)
    if args.answer_prompter:
        outputs_reasoning = outputs
        inputs = tokenizer([prompt + outputs_reasoning + ' ###\nANSWER:'])

        input_ids = torch.as_tensor(inputs.input_ids).cuda()

        keywords = ['###']
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images,
                do_sample=True,
                temperature=0.7,
                max_new_tokens=64,
                stopping_criteria=[stopping_criteria])

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] Sample {i}: {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]

        try:
            index = outputs.index(conv.sep)
        except ValueError:
            outputs += conv.sep
            index = outputs.index(conv.sep)

        outputs = outputs[:index].strip()
        outputs = outputs_reasoning + '\n The answer is ' + outputs

        import pdb; pdb.set_trace()
        print(outputs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mm-projector", type=str, default=None)
    parser.add_argument("--vision-tower", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="simple")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--prompt_type", type=str, default="PROMPT1")
    parser.add_argument("--img_path", type=str, default=None, required=True)
    parser.add_argument("--model_path", type=str, default=None, required=True)
    args = parser.parse_args()

    eval_model(args)
    
#### SETTING UP:
# Follow the steps at https://github.com/microsoft/LLaVA-Med?tab=readme-ov-file#model-download to download and set up the checkpoints
# Clone the LLaVA-Med repository and place this file in llava/eval/

#### USAGE:
# python llava_med_inference.py --img_path <path_to_image> --model_path <path_to_checkpoint>