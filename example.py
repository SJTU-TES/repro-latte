from model import LatteT2V
from t2v import Text2Video


pretrained_model_path = "sjtusora/pretrained/latte"
video_length = 16
text_prompt = [
    'Yellow and black tropical fish dart through the sea.',
    'An epic tornado attacking above aglowing city at night.',
    'Slow pan upward of blazing oak fire in an indoor fireplace.',
    'a cat wearing sunglasses and working as a lifeguard at pool.',
    'Sunset over the sea.',
    'A dog in astronaut suit and sunglasses floating in space.',
]

transformer_model = LatteT2V.from_pretrained_2d(
    pretrained_model_path, 
    subfolder="transformer", 
    video_length=video_length
)

t2v_model = Text2Video(
    pretrained_model_path=pretrained_model_path,
    t2v_ckpt_path="sjtusora/pretrained/latte/t2v.pt",
    transformer_model=transformer_model,
    fp16=True
)

t2v_model.generate(
    text_prompt=text_prompt
)