from tqdm import tqdm
import torch
from typing import Union, List, Optional, Callable
from torchvision import transforms as tfms
from PIL import Image, ImageOps
import numpy as np
from utils import *
import torch.nn as nn
from torchvision.utils import save_image

to_pil = tfms.ToPILImage()

class AttnController:
        def __init__(self, post = False, mask = None,switch=True, criteria = 'MSE', target_depth = [256, 64]) -> None:
            self.attn_probs = []
            self.logs = []
            self.post = post
            self.rand_target = None
            self.rand_targets = None
            self.lamb = 0.5
            self.masks = {'origin':mask}
            self.targets = []
            self.sources = []
            self.m_name = []
            self.criteria_name = criteria
            self.target_depth = target_depth

            if criteria == 'MSE':
                self.criteria = nn.MSELoss()
            elif criteria == 'COS':
                self.criteria = nn.CosineSimilarity(dim = -1, eps=0.00001)
            self.switch = switch
            self.target_hidden = None
        def register_mask(self, hidden_shape):
            if hidden_shape not in self.masks:
                w_h = int(np.sqrt(hidden_shape))
               

                new_mask = torch.nn.functional.interpolate(self.masks["origin"], size=(w_h, w_h))
                new_mask = new_mask.flatten().unsqueeze(0).unsqueeze(2)

                if self.criteria_name == 'COS':
                    new_mask[new_mask==0] = 0.01
                
                self.masks[hidden_shape] = new_mask

        def __call__(self, hidden, m_name):
            if self.switch:
                hidden = hidden.clone()

                uncond, cond = hidden.chunk(2) 
                b, h, c = uncond.shape

                self.register_mask(h)
                if h in self.target_depth:
                    self.sources.append(uncond)
                    self.targets.append(cond)
                    self.m_name.append(m_name)
                    
        def set_concept_positions(self, concept_positions):
            self.concept_positions = concept_positions

        def loss(self, loss_mask, loss_depth):
            losses = 0
            cur_loss = 0
            if self.target_hidden is not None:
                source = self.target_hidden
            else:
                source = self.sources
            
            for i, (a, b) in enumerate(zip(source, self.targets)):
                ba, h, c = a.shape
                loss = 0
                if loss_mask:
                    a = a * self.masks[h]
                    b = b * self.masks[h]
                if h in loss_depth:
                    loss = - self.criteria(a.detach(), b)
                    losses += loss
                    
                cur_loss += loss
                if i % 16 == 15:
                    cur_loss = 0
            self.sources = []
            self.targets = []
            self.m_name = []
            return losses
            
    

        def zero_attn_probs(self):
            self.sources = []
            self.targets = []
            
class MyCrossAttnProcessor:
    def __init__(self, attn_controller: "AttnController", module_name,post= False) -> None:
        self.attn_controller = attn_controller
        self.module_name = module_name
        self.post = False
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size=batch_size)

        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query)

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
    
        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        hidden_states = torch.bmm(attention_probs, value)

        hidden_states = attn.batch_to_head_dim(hidden_states)
    
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        self.attn_controller(hidden_states, self.module_name)
        return hidden_states

def attack_forward(
        self,
        text_embeddings,
        masked_image: Union[torch.FloatTensor, Image.Image],
        mask: Union[torch.FloatTensor, Image.Image],
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        random_t: torch.tensor = None,
    ):
        num_channels_latents = self.vae.config.latent_channels
        latents_shape = (1 , num_channels_latents, height // 8, width // 8)
        latents = torch.randn(latents_shape, device=self.device, dtype=text_embeddings.dtype)

        mask = torch.nn.functional.interpolate(mask, size=(height // 8, width // 8))
        mask = torch.cat([mask] * 2)

        masked_image_latents = self.vae.encode(masked_image).latent_dist.sample()
        masked_image_latents = 0.18215 * masked_image_latents
        masked_image_latents = torch.cat([masked_image_latents] * 2)

        self.scheduler.set_timesteps(num_inference_steps)
        timesteps_tensor = self.scheduler.timesteps.to(self.device)
        timesteps_tensor = random_t

        for i, t in enumerate(timesteps_tensor):
            
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
        return latents, [noise_pred_text, noise_pred_uncond]


def get_grad(cur_mask, cur_masked_image, text_embeddings, pipe, attn_controller, loss_depth, random_t,mask64, pixel_loss=0, loss_mask = False, **kwargs):
    torch.set_grad_enabled(True)
    cur_mask = cur_mask.clone()
    cur_masked_image = cur_masked_image.clone()
    cur_mask.requires_grad = False
    cur_masked_image.requires_grad_()

    image_nat, latents = attack_forward(pipe,mask=cur_mask,
                               masked_image=cur_masked_image,
                               text_embeddings = text_embeddings,
                               random_t = random_t,)

    loss_value = attn_controller.loss(loss_mask, loss_depth)


    loss = loss_value
    
    grad = torch.autograd.grad(loss, [cur_masked_image])[0] * (1 - cur_mask)
        
    return grad, loss_value.item(),0,0# loss_pixel.item(), image_nat.data.cpu()

def get_random_t(t_schedule, t_schedule_bound):
    result = np.empty((0), int)
    for i in t_schedule:
        cur_t = np.clip(np.random.normal(i, 6), i-t_schedule_bound, i+t_schedule_bound)
        cur_t = int(cur_t)
        result = np.append(result, cur_t)
    return torch.tensor(result)


def get_random_emb(embs):
    rand_pos = np.random.randint(0, len(embs))
    return embs[rand_pos]


def disrupt(cur_mask, X, text_embeddings, step_size, iters, eps, clamp_min, clamp_max, attn_controller, pipe, t_schedule, t_schedule_bound, pixel_loss = False, loss_depth = [64], loss_mask = False,infer_unet=None, grad_reps = 5, target_image = 0, **kwargs):
    X_adv = X.clone()
    iterator = tqdm(range(iters))
    total_losses = []
    cur_mask64 = torch.nn.functional.interpolate(cur_mask, size=(64, 64)).detach()
    for j in iterator:
        if j in kwargs["inter_print"]:
            infer_dict = kwargs["infer_dict"]
            with torch.no_grad():
                inter_adv = (X_adv / 2 + 0.5).clamp(0, 1)
                inter_adv = to_pil(inter_adv[0]).convert("RGB")
                inter_adv = recover_image(inter_adv, infer_dict["init_image"], infer_dict["mask"],background=True)
                if infer_unet is not None:
                    temp = pipe.unet
                    pipe.unet = infer_unet
                image_nat = pipe(prompt=infer_dict["prompt"], 
                        image=inter_adv, 
                        mask_image=infer_dict["mask"], 
                        eta=1,
                        num_inference_steps=infer_dict["num_inference_steps"],
                        guidance_scale=infer_dict["guidance_scale"],
                        strength=infer_dict["strength"],
                        generator=torch.Generator(device='cuda').manual_seed(1003),
                        ).images[0]
                inter_adv = recover_image(image_nat, infer_dict["init_image"], infer_dict["mask"])
                inter_adv.save(f"{infer_dict['path']}/{infer_dict['prefix']}_{infer_dict['prompt']}_{j}.png")
                if infer_unet is not None:
                    pipe.unet = temp
            attn_controller.zero_attn_probs()
        random_t = get_random_t(t_schedule, t_schedule_bound)
        all_grads = []
        value_losses = []
        pixel_losses = []
        text_embed = get_random_emb(text_embeddings)
        for i in range(grad_reps):

            c_grad, loss_value, loss_pixel, last_image = get_grad(cur_mask, X_adv, text_embed, attn_controller=attn_controller, pipe=pipe, pixel_loss = pixel_loss, loss_depth = loss_depth, loss_mask=loss_mask, random_t = random_t, mask64=cur_mask64,**kwargs)
            all_grads.append(c_grad)
            value_losses.append(loss_value)
            pixel_losses.append(loss_pixel)
            attn_controller.zero_attn_probs()   

        grad = torch.stack(all_grads).mean(0)
        cur_val_loss = np.mean(value_losses)
        cur_pix_loss = np.mean(pixel_losses)
        total_losses.append([cur_val_loss,cur_pix_loss])

        iterator.set_description_str(f'Loss: {cur_val_loss:.5f}')

        l = len(X.shape) - 1
        grad_norm = torch.norm(grad.detach().reshape(grad.shape[0], -1), dim=1).view(-1, *([1] * l))
        grad_normalized = grad.detach() / (grad_norm + 1e-10)

        actual_step_size = step_size*(1-j/iters)
        X_adv = X_adv - grad_normalized * actual_step_size

        d_x = X_adv - X.detach()
        d_x_norm = torch.renorm(d_x, p=2, dim=0, maxnorm=eps)
        X_adv.data = torch.clamp(X + d_x_norm, clamp_min, clamp_max)        
        
    torch.cuda.empty_cache()

    return X_adv, last_image, total_losses