from tqdm import tqdm
import torch
from typing import Union, List, Optional, Callable
from torchvision import transforms as tfms

from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
import random

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch


from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import prepare_mask_and_masked_image
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers import AsymmetricAutoencoderKL
import PIL
from torchvision.utils import save_image

to_tensor_tfm = tfms.ToTensor()
totensor = tfms.ToTensor()
topil = tfms.ToPILImage()

def save_test_img(img, file_name):
    img_save_dir = './results/'
    x_all = torch.cat([img])
    grid = make_grid(x_all, nrow=10)
    save_image(grid, img_save_dir + file_name)
    print(img.shape)
    print('saved image at ' + img_save_dir + file_name)

    """ visualizing mask and masked image
    diff = cur_masked_image.cpu()
    minn = diff.min()
    max = diff.max()

    normalized_data = 255 * (diff - minn) / (max - minn)
    Image.fromarray(normalized_data.numpy().astype('uint8')[0].transpose(1,2,0))
    """
def torch_to_numpy(image) -> List["PIL_IMAGE"]:
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    return image

def repeat_expand(initial_tensor, desired_length): 
    # Calculate the number of times each element should be repeated
    repeats = desired_length // len(initial_tensor)

    # Calculate the remainder for any remaining positions
    remainder = desired_length % len(initial_tensor)

    initial_tensor = initial_tensor.cpu().numpy()
    # Create the final tensor with evenly spaced values
    final_tensor = np.repeat(initial_tensor, repeats,axis=0)
    final_tensor = np.append(final_tensor, initial_tensor[:remainder],axis=0)

    final_tensor = torch.tensor(final_tensor).cuda().half()
    return final_tensor

def pil_to_latent(pipe,input_im):
  # Single image -> single latent in a batch (so size 1, 4, 64, 64)
  with torch.no_grad():
    latent = pipe.vae.encode(to_tensor_tfm(input_im).to(pipe.vae.dtype).unsqueeze(0).to(pipe.unet.device)*2-1) # Note scaling
  return 0.18215 * latent.latent_dist.mode() # or .mean or .sample

def tensor_to_latent(pipe,input_im):
  # Single image -> single latent in a batch (so size 1, 4, 64, 64)
  with torch.no_grad():
    latent = pipe.vae.encode(input_im.to(pipe.unet.device)*2-1) # Note scaling
  return 0.18215 * latent.latent_dist.mode() # or .mean or .sample



def text_embedding(pipe, prompt):
    text_inputs = pipe.tokenizer(
                prompt,
                padding="max_length",
                max_length=pipe.tokenizer.model_max_length,
                return_tensors="pt",
            )
    text_input_ids = text_inputs.input_ids
    text_embeddings = pipe.text_encoder(text_input_ids.to(pipe.device))[0]

    uncond_tokens = [""]
    max_length = text_input_ids.shape[-1]
    uncond_input = pipe.tokenizer(
        uncond_tokens,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(pipe.device))[0]
    seq_len = uncond_embeddings.shape[1]

    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    text_embeddings = text_embeddings.detach()

    return text_embeddings
def single_text_embedding(pipe, prompt):
    text_inputs = pipe.tokenizer(
                prompt,
                padding="max_length",
                max_length=pipe.tokenizer.model_max_length,
                return_tensors="pt",
            )
    text_input_ids = text_inputs.input_ids
    text_embeddings = pipe.text_encoder(text_input_ids.to(pipe.device))[0]

    text_embeddings = text_embeddings.detach()

    return text_embeddings
def sample(
        self,
        text_embeddings,
        masked_images: Union[torch.FloatTensor, Image.Image],
        mask: Union[torch.FloatTensor, Image.Image],
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        source_latent = None,
        current_t : int = None,
        all_latents= False
    ):
        
        mask = torch.nn.functional.interpolate(mask, size=(height // 8, width // 8))
        mask = torch.cat([mask] * 2)

        self.scheduler.set_timesteps(num_inference_steps)
        # timesteps_tensor = [current_t-i for i in range(num_inference_steps)]
        timesteps =  self.scheduler.timesteps
        if source_latent is not None:
            latents = source_latent
        latent_list = []
        for i, t in enumerate(timesteps):
            #print(t)
            masked_image = masked_images[i][None]
            masked_image_latents = self.vae.encode(masked_image).latent_dist.sample()
            masked_image_latents = 0.18215 * masked_image_latents
            masked_image_latents = torch.cat([masked_image_latents] * 2)

            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = self.scheduler.step(noise_pred, t, latents, eta=eta).prev_sample
            if all_latents:
                latent_list.append(latents)
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        if all_latents:
            return image, latent_list
        else:
            return image, None
        


def recover_image(image, init_image, mask, background=False):
    image = totensor(image)
    mask = totensor(mask)
    init_image = totensor(init_image)
    if background:
        result = mask * init_image + (1 - mask) * image
    else:
        result = mask * image + (1 - mask) * init_image
    return topil(result)

def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0

def prepare_mask_and_masked(image, mask):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)
    masked_image = image * (mask < 0.5)

    return mask, masked_image

def load_prompt(path):
    prompts = []
    with open(path, 'r') as f:
        for line in f:
            prompts.append(line)
    return prompts

def prepare_tensor(image):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

def prepare_image(image):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    return image[0]

def decode_image(self, latents: torch.FloatTensor, **kwargs) -> List["PIL_IMAGE"]:
    scaled_latents = 1 / 0.18215 * latents
    image = [
        self.vae.decode(scaled_latents[i : i + 1]).sample for i in range(len(latents))
    ]
    image = torch.cat(image, dim=0)
    return image
# def torch_to_numpy(self, image) -> List["PIL_IMAGE"]:
#     image = (image / 2 + 0.5).clamp(0, 1)
#     image = image.cpu().permute(0, 2, 3, 1).numpy()
#     return image
def latents_to_imgs(pipe,latents):
    x = decode_image(pipe,latents)
    x = torch_to_numpy(pipe,x.detach())
    x = pipe.numpy_to_pil(x)
    return x

def get_text_embedding(self, prompt):
    text_input_ids = self.tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=self.tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids
    text_embeddings = self.text_encoder(text_input_ids.to(self.device))[0]
    return text_embeddings

def load_img(path, target_size=512):
    """Load an image, resize and output -1..1"""
    image = Image.open(path).convert("RGB")

    tform = tfms.Compose(
        [
            tfms.Resize(target_size),
            tfms.CenterCrop(target_size),
            tfms.ToTensor(),
        ]
    )
    image = tform(image)
    return 2.0 * image - 1.0

def get_image_latents(self, image, sample=True, rng_generator=None):
    encoding_dist = self.vae.encode(image).latent_dist
    if sample:
        encoding = encoding_dist.sample(generator=rng_generator)
    else:
        encoding = encoding_dist.mode()
    latents = encoding * 0.18215
    return latents

def prepare_mask_and_masked2(image, mask, no_mask=False, inverted= False):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    if inverted:
        mask[mask >= 1] = 2
        mask[mask <= 1] = 1
        mask[mask == 2] = 0
    if no_mask:
        mask[mask>=0] = 1
    mask = torch.from_numpy(mask)
    masked_image = image * (mask < 0.5)

    return mask, masked_image,image