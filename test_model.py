r"""
from diffusers import StableDiffusionPipeline
import torch
from diffusers import DDIMScheduler
from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline

model_path = "/home/xiaoguzai/.cache/modelscope/hub/langboat/Guohua-Diffusion"
prompt = "a cute girl, blue eyes, brown hair"
torch.manual_seed(123123123)

pipe = pipeline(task=Tasks.text_to_image_synthesis,
                model='langboat/Guohua-Diffusion',
                model_revision='v1.0')
# def dummy(images, **kwargs):
#     return images, False
# pipe.safety_checker = dummy
#images = pipe(prompt, width=512, height=512, num_inference_steps=30, num_images_per_prompt=3).images
images = pipe({'text':prompt})
print(images)
for i, image in enumerate(images):
    image.save(f"test-{i}.png")
"""

r"""
from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline
import cv2

pipe = pipeline(task=Tasks.text_to_image_synthesis,
                model='/home/xiaoguzai/.cache/modelscope/hub/langboat/Guohua-Diffusion')

prompt = '好看的女孩'
output = pipe({'text': prompt})
cv2.imwrite('result.png', output['output_imgs'][0])
"""

r"""
from modelscope.pipelines import pipeline
from modelscope.outputs import OutputKeys

p = pipeline('text-to-video-synthesis', 'damo/text-to-video-synthesis')
test_text = {
        'text': 'A panda eating bamboo on a rock.',
    }
output_video_path = p(test_text, output_video='./output.mp4')[OutputKeys.OUTPUT_VIDEO]
print('output_video_path:', output_video_path)
"""
from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline
import cv2
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
pipeline = StableDiffusionPipeline.from_pretrained(
                "/home/xiaoguzai/代码/现在正在学习的-dreambooth-for-diffusion-main/new_model",
                torch_dtype=torch.float32,
                safety_checker=None,
                revision=args.revision,
            )

prompt = '愤怒的狗'
output = pipe({'text': prompt})
cv2.imwrite('result.png', output['output_imgs'][0])