import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Image preprocessing function
def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

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


def main(image_path, prompt, model_path):
    # Defining generation configurations
    generation_config = dict(max_new_tokens=1024, do_sample=False)

    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()

    response = model.chat(tokenizer, pixel_values, prompt, generation_config)

    print(response)


if __name__ == "__main__":
    # Example usage:
    main(
        image_path='/path/to/fundus_image',
        # prompt="Describe the fundus image by combining positional information. Please express the organs or lesions in the image in the form of bounding boxes, \"<ref>...</ref><box>[[x1, y1, x2, y2],...]</box>\".",
        # prompt="你的任务是定位眼底彩照中视盘的位置，用边界框的形式表示，[x1, y1, x2, y2]，所有的值都归一化为0到999之间的整数。这些值分别对应左上角的x，左上角的y，右下角的x和右下角的y。请使用这种格式用边界框表示区域：\"<ref>视盘</ref><box>[[x1, y1, x2, y2],...]</box>\"。注意，用户只需要你给出定位坐标。",
        prompt="Your task is to locate the hard exudate in the fundus color photograph, and represent it in the form of a bounding box, [x1, y1, x2, y2], where all values are normalized to integers between 0 and 999. These values correspond to the x of the upper left corner, the y of the upper left corner, the x of the lower right corner, and the y of the lower right corner. Please use this format to represent the region with a bounding box: \"<ref>Hard exudate</ref><box>[[x1, y1, x2, y2],...]</box>\". Note that the user only needs you to give the location coordinates.",
        # prompt="Please describe the fundus image in detail and give a preliminary diagnostic conclusion of the fundus image.",
        model_path='/path/to/model'
    )