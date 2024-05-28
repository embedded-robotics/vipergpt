from __future__ import annotations

import numpy as np
import re
import torch
from dateutil import parser as dateparser
from PIL import Image
from rich.console import Console
from torchvision import transforms
from torchvision.ops import box_iou
from typing import Union, List
from word2number import w2n

from utils import show_single_image, load_json
from vision_processes import forward, config

console = Console(highlight=False)


class ImagePatch:
    """A Python class containing a crop of an image centered around a particular object, as well as relevant
    information.
    Attributes
    ----------
    cropped_image : array_like
        An array-like of the cropped image taken from the original image.
    left : int
        An int describing the position of the left border of the crop's bounding box in the original image.
    lower : int
        An int describing the position of the bottom border of the crop's bounding box in the original image.
    right : int
        An int describing the position of the right border of the crop's bounding box in the original image.
    upper : int
        An int describing the position of the top border of the crop's bounding box in the original image.

    Methods
    -------
    find(object_name: str)->List[ImagePatch]
        Returns a list of new ImagePatch objects containing crops of the image centered around any objects found in the
        image matching the object_name.
    exists(object_name: str)->bool
        Returns True if the object specified by object_name is found in the image, and False otherwise.
    verify_property(property: str)->bool
        Returns True if the property is met, and False otherwise.
    best_text_match(option_list: List[str], prefix: str)->str
        Returns the string that best matches the image.
    simple_query(question: str=None)->str
        Returns the answer to a basic question asked about the image. If no question is provided, returns the answer
        to "What is this?".
    compute_depth()->float
        Returns the median depth of the image crop.
    crop(left: int, lower: int, right: int, upper: int)->ImagePatch
        Returns a new ImagePatch object containing a crop of the image at the given coordinates.
    """

    def __init__(self, image: Union[Image.Image, torch.Tensor, np.ndarray], left: int = None, lower: int = None,
                 right: int = None, upper: int = None, parent_left=0, parent_lower=0, queues=None,
                 parent_img_patch=None):
        """Initializes an ImagePatch object by cropping the image at the given coordinates and stores the coordinates as
        attributes. If no coordinates are provided, the image is left unmodified, and the coordinates are set to the
        dimensions of the image.

        Parameters
        -------
        image : array_like
            An array-like of the original image.
        left : int
            An int describing the position of the left border of the crop's bounding box in the original image.
        lower : int
            An int describing the position of the bottom border of the crop's bounding box in the original image.
        right : int
            An int describing the position of the right border of the crop's bounding box in the original image.
        upper : int
            An int describing the position of the top border of the crop's bounding box in the original image.

        """

        if isinstance(image, Image.Image):
            # image = transforms.ToTensor()(image)
            image = image
        elif isinstance(image, np.ndarray):
            image = torch.tensor(image).permute(1, 2, 0)
        elif isinstance(image, torch.Tensor) and image.dtype == torch.uint8:
            image = image / 255

        if left is None and right is None and upper is None and lower is None:
            self.cropped_image = image
            self.left = 0
            self.lower = 0
            if isinstance(image, Image.Image):
                self.right = image.size[0]  # width
                self.upper = image.size[1]  # height
            else:
                self.right = image.shape[2]  # width
                self.upper = image.shape[1]  # height
        else:
            if isinstance(image, Image.Image):
                self.cropped_image = image.crop((left, upper, right, lower))
            else:
                self.cropped_image = image[:, image.shape[1]-upper:image.shape[1]-lower, left:right]
            self.left = left + parent_left
            self.upper = upper + parent_lower
            self.right = right + parent_left
            self.lower = lower + parent_lower

        if isinstance(image, Image.Image):
            self.height = self.cropped_image.size[1]
            self.width = self.cropped_image.size[0]
        else:
            self.height = self.cropped_image.shape[1]
            self.width = self.cropped_image.shape[2]

        self.cache = {}
        self.queues = (None, None) if queues is None else queues

        self.parent_img_patch = parent_img_patch

        self.horizontal_center = (self.left + self.right) / 2
        self.vertical_center = (self.lower + self.upper) / 2

        if isinstance(image, Image.Image):
            if self.cropped_image.size[0] == 0 or self.cropped_image.size[0] == 0:
                raise Exception("ImagePatch has no area")
        else:
            if self.cropped_image.shape[1] == 0 or self.cropped_image.shape[2] == 0:
                raise Exception("ImagePatch has no area")

        self.possible_options = load_json('./useful_lists/possible_options.json')
        
    def forward(self, model_name, *args, **kwargs):
        return forward(model_name, *args, queues=self.queues, **kwargs)

    @property
    def original_image(self):
        if self.parent_img_patch is None:
            return self.cropped_image
        else:
            return self.parent_img_patch.original_image


    def find_histo_patches(self) -> list[ImagePatch]:
        """Return patches of region of interest where the nucleis are densly populated in the image.
        Returns
        -------
        List[ImagePatch]
            a list of ImagePatch objects containing image crops densely populated with pathology information within the original crop
        """

        all_object_coordinates = self.forward('histocartography', self.cropped_image)
        return [self.crop(coordinates[0], coordinates[3], coordinates[2], coordinates[1]) for coordinates in all_object_coordinates]
        
    def llava_med_general_query(self, question: str) -> str:
        """Returns the answer to a pathology question asked about the image. 
        The questions are about basic perception, and are not meant to be used for complex reasoning
        or external knowledge.
        Parameters
        -------
        question : str
            A string describing the question to be asked.
        """
        return self.forward('llava_med', self.cropped_image, question)

    def llava_med_identify_location(self) -> str:
        """
        Determining the specific organ or tissue site under examination. Identifying the type of gland present.
        -------
        Returns:
            str: A description of the location within the body.
        """
        query = "Determining the specific organ or tissue site under examination. Identifying the type of gland present."
        return self.forward('llava_med', self.cropped_image, query)


    def llava_med_identify_abnormality(self) -> str:
        """
        Identify any abnormalities or lesions present. Look for areas of calcification or nodular densities within the soft tissue.
        -----
        Returns:
            str: A description of the abnormality or lesion.
        """
        query = "Identify any abnormalities or lesions present. Look for areas of calcification or nodular densities within the soft tissue."
        return self.forward('llava_med', self.cropped_image, query)

       # def llava_med_postmortem(self) -> bool:
    #     """
    #     Evaluating whether any observed changes are a result of postmortem processes, which may render diagnostic interpretations unreliable.
    #     """
    #     query = "Examine whether any observed changes are a result of postmortem processes. Answer yes or no."
    #     ans = self.forward('llava_med', self.cropped_image, query)
    #     return "yes" in ans.lower()
        
    def llava_med_exists(self, object_name: str) -> bool:
        """Returns True if the object specified by object_name is found in the image, and False otherwise.
        Parameters
        -------
        object_name : str
            A string describing the name of the object to be found in the image.
        """
        query = f"Is {object_name} presented in this image? Answer yes or no."
        ans = self.forward('llava_med', self.cropped_image, query)
        return "yes" in ans
    
    def llava_med_verify_property(self, object_name: str, attribute: str) -> bool:
        """
        Returns True if the object possesses the property, and False otherwise.
        Differs from 'exists' in that it presupposes the existence of the object specified by object_name, instead
        checking whether the object possesses the property.
        This calls a specialized model for pathology images. It is not a general purpose model and will fail on
        non-pathology images.
        Parameters
        -------
        object_name : str
            A string describing the name of the object to be found in the image.
        attribute : str
            A string describing the property to be checked.
        """
        attribute_name = f"{attribute} {object_name}"
        query = f"Is {object_name} being {attribute_name}? Answer yes or no."
        ans = self.forward('llava_med', self.cropped_image, query)
        return "yes" in ans

    def medical_expert_query(self, question: str, long_answer: bool = True) -> str:
        """Returns the answer to a pathology question. 
        """
        
        question = f"You're a pathologist, please answer the following question: {question}."
        return llm_query(question, None, long_answer)

    def medical_expert_summarization(self, context: str, long_answer: bool = True) -> str:
        """Returns the summarization of the given context.
        """
        
        question = f"You're a pathologist, please summarize the following context into a reasonable answer: {context}."
        return llm_query(question, None, long_answer)

    def crop(self, left: int, lower: int, right: int, upper: int) -> ImagePatch:
        """Returns a new ImagePatch containing a crop of the original image at the given coordinates.
        Parameters
        ----------
        left : int
            the position of the left border of the crop's bounding box in the original image
        lower : int
            the position of the bottom border of the crop's bounding box in the original image
        right : int
            the position of the right border of the crop's bounding box in the original image
        upper : int
            the position of the top border of the crop's bounding box in the original image

        Returns
        -------
        ImagePatch
            a new ImagePatch containing a crop of the original image at the given coordinates
        """
        # make all inputs ints
        left = int(left)
        lower = int(lower)
        right = int(right)
        upper = int(upper)

        if config.crop_larger_margin:
            left = max(0, left - 10)
            lower = max(0, lower - 10)
            right = min(self.width, right + 10)
            upper = min(self.height, upper + 10)

        return ImagePatch(self.cropped_image, left, lower, right, upper, self.left, self.lower, queues=self.queues,
                          parent_img_patch=self)

    def overlaps_with(self, left, lower, right, upper):
        """Returns True if a crop with the given coordinates overlaps with this one,
        else False.
        Parameters
        ----------
        left : int
            the left border of the crop to be checked
        lower : int
            the lower border of the crop to be checked
        right : int
            the right border of the crop to be checked
        upper : int
            the upper border of the crop to be checked

        Returns
        -------
        bool
            True if a crop with the given coordinates overlaps with this one, else False
        """
        return self.left <= right and self.right >= left and self.lower <= upper and self.upper >= lower

    # def llm_query(self, question: str, long_answer: bool = True) -> str:
    #     return llm_query(question, None, long_answer)

    def print_image(self, size: tuple[int, int] = None):
        show_single_image(self.cropped_image, size)

    def __repr__(self):
        return "ImagePatch({}, {}, {}, {})".format(self.left, self.lower, self.right, self.upper)


def best_image_match_plip(patch_list: list[ImagePatch], prompt: str) -> ImagePatch:
    """Returns the image that best matches the string. This calls a specialized model for pathology images. It is not a general purpose model and will fail on
    non-pathology images.
    Parameters
    -------
    patch_list : ImagePatch
        a list of ImagePatch objects to be compared against the prompt
    prompt : str
        A string to match against different images
    """

    model_name = config.plip_model
    selected = patch_list[0].forward(model_name, [p.cropped_image for p in patch_list], prompt, task='compare')
    return patch_list[selected]


def best_image_match(list_patches: list[ImagePatch], content: List[str], return_index: bool = False) -> \
        Union[ImagePatch, None]:
    """Returns the patch most likely to contain the content.
    Parameters
    ----------
    list_patches : List[ImagePatch]
    content : List[str]
        the object of interest
    return_index : bool
        if True, returns the index of the patch most likely to contain the object

    Returns
    -------
    int
        Patch most likely to contain the object
    """
    if len(list_patches) == 0:
        return None

    model = config.best_match_model

    scores = []
    for cont in content:
        if model == 'clip':
            res = list_patches[0].forward(model, [p.cropped_image for p in list_patches], cont, task='compare',
                                          return_scores=True)
        else:
            res = list_patches[0].forward(model, [p.cropped_image for p in list_patches], cont, task='score')
        scores.append(res)
    scores = torch.stack(scores).mean(dim=0)
    scores = scores.argmax().item()  # Argmax over all image patches

    if return_index:
        return scores
    return list_patches[scores]


def distance(patch_a: Union[ImagePatch, float], patch_b: Union[ImagePatch, float]) -> float:
    """
    Returns the distance between the edges of two ImagePatches, or between two floats.
    If the patches overlap, it returns a negative distance corresponding to the negative intersection over union.
    """

    if isinstance(patch_a, ImagePatch) and isinstance(patch_b, ImagePatch):
        a_min = np.array([patch_a.left, patch_a.lower])
        a_max = np.array([patch_a.right, patch_a.upper])
        b_min = np.array([patch_b.left, patch_b.lower])
        b_max = np.array([patch_b.right, patch_b.upper])

        u = np.maximum(0, a_min - b_max)
        v = np.maximum(0, b_min - a_max)

        dist = np.sqrt((u ** 2).sum() + (v ** 2).sum())

        if dist == 0:
            box_a = torch.tensor([patch_a.left, patch_a.lower, patch_a.right, patch_a.upper])[None]
            box_b = torch.tensor([patch_b.left, patch_b.lower, patch_b.right, patch_b.upper])[None]
            dist = - box_iou(box_a, box_b).item()

    else:
        dist = abs(patch_a - patch_b)

    return dist


def bool_to_yesno(bool_answer: bool) -> str:
    """Returns a yes/no answer to a question based on the boolean value of bool_answer.
    Parameters
    ----------
    bool_answer : bool
        a boolean value

    Returns
    -------
    str
        a yes/no answer to a question based on the boolean value of bool_answer
    """
    return "yes" if bool_answer else "no"


def llm_query(query, context=None, long_answer=True, queues=None):
    """Answers a text question using GPT-3. The input question is always a formatted string with a variable in it.

    Parameters
    ----------
    query: str
        the text question to ask. Must not contain any reference to 'the image' or 'the photo', etc.
    """
    if long_answer:
        return forward(model_name='gpt3_general', prompt=query, queues=queues)
    else:
        return forward(model_name='gpt3_qa', prompt=[query, context], queues=queues)


def coerce_to_numeric(string, no_string=False):
    """
    This function takes a string as input and returns a numeric value after removing any non-numeric characters.
    If the input string contains a range (e.g. "10-15"), it returns the first value in the range.
    # TODO: Cases like '25to26' return 2526, which is not correct.
    """
    if any(month in string.lower() for month in ['january', 'february', 'march', 'april', 'may', 'june', 'july',
                                                 'august', 'september', 'october', 'november', 'december']):
        try:
            return dateparser.parse(string).timestamp().year
        except:  # Parse Error
            pass

    try:
        # If it is a word number (e.g. 'zero')
        numeric = w2n.word_to_num(string)
        return numeric
    except ValueError:
        pass

    # Remove any non-numeric characters except the decimal point and the negative sign
    string_re = re.sub("[^0-9\.\-]", "", string)

    if string_re.startswith('-'):
        string_re = '&' + string_re[1:]

    # Check if the string includes a range
    if "-" in string_re:
        # Split the string into parts based on the dash character
        parts = string_re.split("-")
        return coerce_to_numeric(parts[0].replace('&', '-'))
    else:
        string_re = string_re.replace('&', '-')

    try:
        # Convert the string to a float or int depending on whether it has a decimal point
        if "." in string_re:
            numeric = float(string_re)
        else:
            numeric = int(string_re)
    except:
        if no_string:
            raise ValueError
        # No numeric values. Return input
        return string
    return numeric
