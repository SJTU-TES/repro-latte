import os
import math
import torch
import imageio
from torch import nn
from utils.scheduler import get_scheduler
from .pipeline_videogen import VideoGenPipeline
from diffusers.models import AutoencoderKL
from transformers import T5EncoderModel, T5Tokenizer


class Text2Video:
    def __init__(
        self,
        pretrained_model_path: str,
        t2v_ckpt_path: str,
        transformer_model: nn.Module,
        # schedule
        sample_method: str="PNDM",
        beta_start: float=0.0001,
        beta_end: float=0.02,
        beta_schedule: str="linear",
        variance_type: str="learned_range",
        # device
        fp16: bool=False,
        device: str="cuda",
    ) -> None:        
        # fp16
        if fp16:
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.float32
        
        # transformer_model
        self.transformer_model = transformer_model.to(device, dtype=self.torch_dtype)
        state_dict = self.load_model(t2v_ckpt_path)
        self.transformer_model.load_state_dict(state_dict)
        
        # vae
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_path, 
            subfolder="vae", 
            torch_dtype=self.torch_dtype
        ).to(device)
        
        # tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(
            pretrained_model_path, 
            subfolder="tokenizer"
        )
        
        # text_encoder
        self.text_encoder = T5EncoderModel.from_pretrained(
            pretrained_model_path, 
            subfolder="text_encoder", 
            torch_dtype=self.torch_dtype
        ).to(device)

        # set eval mode
        self.transformer_model.eval()
        self.vae.eval()
        self.text_encoder.eval()
    
        # scheduler
        self.scheduler = get_scheduler(
            name=sample_method,
            pretrained_model_path=pretrained_model_path,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            variance_type=variance_type
        )
        
        # pipeline
        self.videogen_pipeline = VideoGenPipeline(
            vae=self.vae, 
            text_encoder=self.text_encoder, 
            tokenizer=self.tokenizer, 
            scheduler=self.scheduler, 
            transformer=self.transformer_model
        ).to(device)
    
    def load_model(self, t2v_ckpt_path: str):
        checkpoint = torch.load(
            t2v_ckpt_path, 
            map_location=lambda storage, loc: storage
        )
        if "ema" in checkpoint:  # supports checkpoints from train.py
            checkpoint = checkpoint["ema"]
        else:
            checkpoint = checkpoint['model']
        return checkpoint
    
    def generate(
        self, 
        text_prompt: list,     
        video_length: int=16,
        image_size: list=[512, 512],
        enable_temporal_attentions: bool=True,
        num_sampling_steps: int=50,
        guidance_scale: float=7.5,
        save_img_path: str="output"
    ):
        # save_path
        if not os.path.exists(save_img_path):
            os.makedirs(save_img_path)
        # generate video
        video_grids = []
        for prompt in text_prompt:
            prompt: str
            print('Processing the ({}) prompt'.format(prompt))
            videos = self.videogen_pipeline(
                prompt, 
                video_length=video_length, 
                height=image_size[0], 
                width=image_size[1], 
                num_inference_steps=num_sampling_steps,
                guidance_scale=guidance_scale,
                enable_temporal_attentions=enable_temporal_attentions,
                num_images_per_prompt=1,
                mask_feature=True,
                enable_vae_temporal_decoder=False
            ).video
            
            try:
                imageio.mimwrite(
                    os.path.join(
                        save_img_path, 
                        prompt.replace(' ', '_') + '_' + 'webv-imageio.mp4'
                    ), 
                    videos[0], 
                    fps=8, 
                    quality=9
                ) # highest quality is 10, lowest is 0
            except:
                print('Error when saving {}'.format(prompt))
            video_grids.append(videos)
        
        # merge
        video_grids = torch.cat(video_grids, dim=0)
        video_grids = self.save_video_grid(video_grids)
        imageio.mimwrite(
            os.path.join(save_img_path, 'merge'+'.mp4'), 
            video_grids, 
            fps=8, 
            quality=5
        )
    
    def save_video_grid(self, video: torch.Tensor, nrow=None):
        b, t, h, w, c = video.shape
        if nrow is None:
            nrow = math.ceil(math.sqrt(b))
        ncol = math.ceil(b / nrow)
        padding = 1
        video_grid = torch.zeros((t, (padding + h) * nrow + padding,
                            (padding + w) * ncol + padding, c), dtype=torch.uint8)
        for i in range(b):
            r = i // ncol
            c = i % ncol
            start_r = (padding + h) * r
            start_c = (padding + w) * c
            video_grid[:, start_r:start_r + h, start_c:start_c + w] = video[i]
        return video_grid

