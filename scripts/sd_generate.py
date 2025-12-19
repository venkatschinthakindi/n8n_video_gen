from diffusers import StableDiffusionPipeline
import torch, argparse, os
parser = argparse.ArgumentParser()
parser.add_argument("--prompt", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32
).to("cpu")

image = pipe(args.prompt, num_inference_steps=15).images[0]
os.makedirs(os.path.dirname(args.out), exist_ok=True)
image.save(args.out)
print("Generated:", args.out)
