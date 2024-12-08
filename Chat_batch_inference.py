import torch
from transformers import AutoTokenizer, AutoModel
import math
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import nvtx
import time 

def split_model():
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = 80

    # Since the first GPU will be used for ViT, allocate only 25% of its share for layers
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))  # Adjust total GPU capacity
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)  # Assign 25% for GPU 0

    # Redistribute remaining layers among other GPUs
    remaining_layers = num_layers - num_layers_per_gpu[0]
    num_layers_per_gpu[1:] = [math.ceil(remaining_layers / (world_size - 1))] * (world_size - 1)

    # Adjust for any rounding differences to ensure total layers match
    layer_cnt = 0
    for i, _ in enumerate(num_layers_per_gpu):
        if i == world_size - 1:  # Last GPU takes the remaining layers
            num_layers_per_gpu[i] = num_layers - layer_cnt
        layer_cnt += num_layers_per_gpu[i]

    # Assign layers to GPUs
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1

    # Map additional model components to GPU 0
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

path = "/workspace/huggingface_cache/hub/models--nvidia--NVLM-D-72B/snapshots/5a57d927ac0ab6b0a96ebc90f5ee7901ddca790d"

device_map = split_model()
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=False,
    trust_remote_code=True,
    device_map=device_map).eval()

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
generation_config = dict(max_new_tokens=1024, do_sample=False)

print(model)
questions = [
    "<image>\nExplain why this meme is funny",
    "<image>\nWhat is the dierence between the left, middle and right object in the image?",
    "<image>\nYou are a helpful driving assistant. In this scene, which lane should I choose and why?",
    "<image>\nWrite code based on the provided pseudo code",
    "<image>\nAccording to the table, explain which food is the most likely cause of the outbreak of food poisoning?",
    "<image>\nWhat percentage of market share does NVIDIA have for data center GPUs in 2023?"



]

images_path = ["/workspace/NVLM/pictures/question1.png","/workspace/NVLM/pictures/question2.png","/workspace/NVLM/pictures/question3.png",
              "/workspace/NVLM/pictures/question4.png","/workspace/NVLM/pictures/question5.png","/workspace/NVLM/pictures/question6.png"              
]


@nvtx.annotate("Pure-Text Conversation one", color="green")  # Annotate with green
def conversation(model, tokenizer, questions, images_path,generation_config,histories):
    pixel_values_list=[]

    for i in range(len(questions)):
        if images_path[i] is not None:
            pixel_values = load_image(images_path[i], max_num=6).to(torch.bfloat16)
            pixel_values_list.append(pixel_values)
        else:
            pixel_values_list.append(None)

    return model.chat_batch(tokenizer, pixel_values_list, questions, generation_config, histories, return_histories=False)



start_time = time.time()
responses= conversation(model, tokenizer, questions, images_path,generation_config,histories=None)
end_time = time.time()
total_time = end_time-start_time
for question, response in zip(questions, responses):
    print(f"User: {question}\nAssistant: {response}\n")

print(f"Total batch exeuction time : {total_time:.2f} seconds")



# inferebce output
# User: <image>
# Explain why this meme is funny
# Assistant: This meme is funny because it uses a humorous comparison to highlight the contrast between two different things. On the left side of the meme, there is a picture of a lynx, which is a wild and fierce animal. On the right side, there is a picture of a domestic cat, which is a common and domesticated pet. The caption "the abstract" on the left side suggests that the lynx represents something abstract or difficult to understand, while the caption "the paper" on the right side suggests that the cat represents something simple and straightforward. The humor comes from the unexpected comparison and the contrast between the two images.

# User: <image>
# What is the dierence between the left, middle and right object in the image?
# Assistant: The left object in the image is a CPU, the middle object is a GPU and the right object is a TPU. The main difference between these three is that a CPU is a general-purpose processor that handles a wide range of tasks, while a GPU is specialized for graphics processing and parallel computing tasks. A TPU, on the other hand, is a specialized processor designed specifically for machine learning and AI tasks.

# User: <image>
# You are a helpful driving assistant. In this scene, which lane should I choose and why?
# Assistant: 'this scene, you should choose the right lane, as indicated by the sign that reads "Autos/BUS & RV." This lane is designated for cars, buses, and recreational vehicles. The other two lanes are closed, as indicated by the signs that read "LANE CLOSED" and the red X symbols.

# User: <image>
# Write code based on the provided pseudo code
# Assistant: : Set smallest number/minimum to first element (index 0) in the list.
# : Look for the smallest number/minimum element in the list.
# : Swap that value with item at index [min].
# : Increment index of [min] to next element.
# : Repeat until last element/list is sorted!

# User: <image>
# According to the table, explain which food is the most likely cause of the outbreak of food poisoning?
# Assistant: : The table shows that the highest percentage of people who ate a particular food and then got sick ate egg sandwiches. This suggests that egg sandwiches are the most likely cause of the outbreak.

# User: <image>
# What percentage of market share does NVIDIA have for data center GPUs in 2023?
# Assistant: 0.92

# Total batch exeuction time : 23.72 seconds