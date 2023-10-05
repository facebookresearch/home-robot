import ast
import enum
import json
import os
import pdb
import random
import time
from typing import Optional

import click
import fire
import numpy as np
import openai
import requests
from llama import Llama
from PIL import Image, ImageDraw

# from pycocotools.coco import COCO
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, LlamaForCausalLM

save_dir = "/private/home/xiaohanzhang/data/webscale_test/"
ckpt_dir = "/checkpoint/arjunmajumdar/weights/llama-2/llama-2-70b-chat/"
tokenizer_path = (
    "/checkpoint/arjunmajumdar/weights/llama-2/llama-2-70b-chat-hf/tokenizer.model"
)

max_seq_len = 512
max_batch_size = 8

num_data_per_task = 30
timestamp = str(time.time()).replace(".", "")
chatgpt_try_times = 5
text_model = SentenceTransformer("all-MiniLM-L6-v2")
openai.api_key = "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"
os.makedirs("datadump", exist_ok=True)
os.makedirs("datadump/images_" + timestamp, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)
os.makedirs(save_dir + "images_" + timestamp, exist_ok=True)

padding = 1.5
task_vocab_size = 100
query_gt_times = 100000000

# label to be explore for each task (plan not found)
neg_ratio = 0.1
partial_pos_ratio = 0.2  # end_recep not found, only support for ovmm-style task for now

object_masks = ["***", "&&&", "$$$"]
tasks = {
    0: "bring me something to drink",
    1: "find ***",
    2: "pickup ***",
    3: "move *** to the &&&",
    4: "move *** from the &&& to the $$$",
    5: "set up the table for dinner",
    6: "bring me something to eat",
}
# only creative tasks are indexed in the following dict
predefined_receps = {
    0: "person",
    5: "table",
    6: "person",
}

gt_keyword = {
    0: "drink",
    5: "utensils",
    6: "eat",
}

# by default is one
max_num_pickable_objects = {
    5: 3,
    0: 2,
    6: 2,
}

# num_task_rephrase = 20 # handle it offline

max_context_length = 20
img_index = 0

with open("/datasets01/lvis/032922/v1/lvis_v1_train.json") as f:
    data = json.load(f)
annotations = data["annotations"]

vocab = []
with open("lvis_household_objects_from_chatgpt.json") as f:
    vocab = json.load(f)
# pdb.set_trace()
templates = {}
for task_template_fname in os.listdir("rephrase_templates/"):
    with open("rephrase_templates/" + task_template_fname) as f:
        templates[int(task_template_fname.split(".")[0])] = f.readlines()

for _, v in predefined_receps.items():
    if v not in vocab:
        print("error: one or more predefined receps are not in vocab.")
        exit()


def ask_chatgpt(content):
    for attempt in range(chatgpt_try_times):
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", messages=[{"role": "user", "content": content}]
            )
        except openai.error.APIError as e:
            # Handle API error here, e.g. retry or log
            print(f"OpenAI API returned an API Error: {e}")
            if attempt < chatgpt_try_times - 1:
                continue
            else:
                exit()
        except openai.error.APIConnectionError as e:
            # Handle connection error here
            print(f"Failed to connect to OpenAI API: {e}")
            if attempt < chatgpt_try_times - 1:
                continue
            else:
                exit()
        except openai.error.RateLimitError as e:
            # Handle rate limit error (we recommend using exponential backoff)
            print(f"OpenAI API request exceeded rate limit: {e}")
            exit()
        break
    return completion["choices"][0]["message"]["content"]


def get_object_image_lvis(cat, vocab):
    if cat not in vocab:
        print("Fatel error in getting object images. Exit.")
        exit()
    global img_index
    random.shuffle(annotations)
    for annotation in annotations:
        if data["categories"][annotation["category_id"] - 1]["name"] == cat:
            break
    for im in data["images"]:
        if im["id"] == annotation["image_id"]:
            img_url = im["coco_url"]
            break
    # /datasets01/COCO/022719/
    im_raw = Image.open("/datasets01/COCO/022719/" + "/".join(img_url.split("/")[-2:]))
    # im_raw = Image.open(requests.get(img_url, stream=True).raw)
    im_w = im_raw.width
    im_h = im_raw.height
    # print(img_url)
    x, y, w, h = annotation["bbox"]
    # x = 0 if (x-(padding-1)*w/2) < 0 else int(x-(padding-1)*w/2)
    # y = 0 if (y-(padding-1)*h/2) < 0 else int(y-(padding-1)*h/2)
    # actual_h = im_h if y+int(h*padding) >= im_h else y+int(h*padding)
    # actual_w = im_w if x+int(w*padding) >= im_w else x+int(w*padding)
    x = 0 if (x - (padding - 1) * w / 2) < 0 else int(x - (padding - 1) * w / 2)
    y = 0 if (y - (padding - 1) * h / 2) < 0 else int(y - (padding - 1) * h / 2)
    y2 = im_h if y + int(h * padding) >= im_h else y + int(h * padding)
    x2 = im_w if x + int(w * padding) >= im_w else x + int(w * padding)
    if y2 == y:
        y2 = y + 1
    if x2 == x:
        x2 = x + 1
    im = np.asarray(im_raw)[y:y2, x:x2]
    image_name = str(img_index) + "_" + cat + ".png"
    Image.fromarray(im).save(save_dir + "images_" + timestamp + "/" + image_name)

    # ImageDraw.Draw(im_raw).polygon(
    #     annotation['segmentation'][0], outline=1, fill=1)
    # image_name = str(img_index)+"_"+cat+"_mask.png"
    # im_raw.save("datadump/images/"+image_name)

    # img = Image.new('L', (im_raw.width, im_raw.height), 255)
    # ImageDraw.Draw(img).polygon(
    #     annotation['segmentation'][0], outline=1, fill=1)
    # image_name = str(img_index)+"_"+cat+"_mask_only.png"
    # img.save("datadump/images/"+image_name)
    # import pdb
    # pdb.set_trace()

    img_index += 1
    return image_name


def wrap_to_plan(obj, recep):
    text = (
        "goto({obj})###pickup({obj})###goto({recep})###placeon({obj}, {recep})".format(
            obj=obj, recep=recep
        )
    )
    return text.split("###")


def wrap_to_plan_ovmm(task_id, images):
    if task_id == 1:
        text = "goto({obj})".format(obj=images[0])
    elif task_id == 2:
        text = "goto({obj})###pickup({obj})".format(obj=images[0])
    elif task_id == 3:
        text = "goto({obj})###pickup({obj})###goto({recep})###placeon({obj}, {recep})".format(
            obj=images[0], recep=images[1]
        )
    elif task_id == 4:
        text = "goto({obj}, {start_recep})###pickup({obj})###goto({recep})###placeon({obj}, {recep})".format(
            obj=images[0], start_recep=images[1], recep=images[2]
        )
    else:
        print("Fatel error: Plan wrapper not implemented. Exit.")
        exit()
    return text.split("###")


def wrap_to_plan_ovmm_partial(task_id, images):
    if task_id == 3:
        text = "goto({obj})###pickup({obj})###explore".format(obj=images[0])
    elif task_id == 4:
        text = "goto({obj}, {start_recep})###pickup({obj})###explore)".format(
            obj=images[0], start_recep=images[1]
        )
    else:
        print("Fatel error: Plan wrapper not implemented. Exit.")
        exit()
    return text.split("###")


def rank_vocab_by_keyword(vocab, keyword):
    emb_vocab = text_model.encode(vocab, convert_to_tensor=True)
    emb_task = text_model.encode(keyword, convert_to_tensor=True)
    scores = util.cos_sim(emb_task, emb_vocab)
    ranked_vocab = sorted(vocab, key=lambda x: scores[0][vocab.index(x)].cpu().numpy())
    return ranked_vocab


def get_household_objects_from_chatgpt(vocab):
    subset = []
    for cat in vocab:
        query = (
            "Is it common to see "
            + cat
            + " in home environment? Only answer yes or no."
        )
        print(query)
        if "yes" in ask_chatgpt(query).lower():
            subset.append(cat)
        with open("lvis_household_objects_from_chatgpt.json", "w") as f:
            json.dump(subset, f)
    return subset


def load_llama2_local(ckpt_dir, tokenizer_path, max_seq_len, max_batch_size):
    return Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )


def ask_llama2_local(generator, content):
    dialogs = [[{"role": "user", "content": content}]]
    print(dialogs)
    max_gen_len = None
    temperature = 0.6
    top_p = 0.9
    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    for dialog, result in zip(dialogs, results):
        return result["generation"]["content"].strip()
        # print(
        #     f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        # )


def load_llama2_hf():
    checkpoint_path = "/checkpoint/arjunmajumdar/weights/llama-2/llama-2-13b-chat-hf"
    print("loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, device_map="auto")
    print("loading tokenizer... done!")

    print("loading model...")
    model = LlamaForCausalLM.from_pretrained(checkpoint_path, device_map="auto")
    print("loading model... done!")
    return tokenizer, model


def ask_llama2_hf(tokenizer, model, prompt, max_new_tokens=8):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generate_ids = model.generate(inputs.input_ids, max_new_tokens=max_new_tokens)
    output = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    answer = " ".join(output[0].strip().split(prompt)[1:])
    if "yes" in answer.lower():
        return "yes"
    else:
        return "no"


def get_gt_objects_from_chatgpt(vocab, task_id):
    target_objects = []
    for candidate_object in vocab:
        # for _ in range(query_gt_times):
        candidate_objects = [candidate_object]
        prompt1 = (
            'can a waiter fulfill the request of "'
            + tasks[task_id]
            + '" using '
            + " and ".join(candidate_objects)
            + "? Only answer yes or no."
        )
        print(prompt1)
        prompt2 = (
            "can "
            + " and ".join(candidate_objects)
            + " be picked up and moved around by a household robot? Only answer yes or no."
        )
        answer1 = ask_chatgpt(prompt1)
        print(answer1)
        time.sleep(1)
        answer2 = ask_chatgpt(prompt2)
        if "yes" in answer1.lower() and "yes" in answer2.lower():
            target_objects.append(candidate_objects)
            # break
    print(target_objects)
    print(len(target_objects))


def get_multiple_gt_objects_from_chatgpt(vocab, task_id, num_gt=30):
    ranked_vocab = rank_vocab_by_keyword(vocab, gt_keyword[task_id])
    vocab = ranked_vocab[-task_vocab_size:]
    target_objects = []
    while len(target_objects) < num_gt:
        num_task_objects = random.randrange(1, max_num_pickable_objects[task_id])
        # for _ in range(query_gt_times):
        candidate_objects = random.sample(vocab, num_task_objects)
        # for i in range(num_task_objects):
        #     candidate_objects[i] = candidate_objects[i].replace('_', ' ')
        prompt1 = (
            'can a waiter fulfill the request of "'
            + tasks[task_id]
            + '" using '
            + " and ".join(candidate_objects)
            + "? Only answer yes or no."
        )
        print(prompt1)
        prompt2 = (
            "can "
            + " and ".join(candidate_objects)
            + " be picked up and moved around by a household robot? Only answer yes or no."
        )
        answer1 = ask_chatgpt(prompt1)
        print(answer1)
        time.sleep(1)
        answer2 = ask_chatgpt(prompt2)
        if "yes" in answer1.lower() and "yes" in answer2.lower():
            target_objects.append(candidate_objects)
        print(target_objects)
        # break
    print(target_objects)
    print(len(target_objects))


@click.command()
@click.option("--task_id", required=True)
def main(task_id):
    # generator = load_llama2_local(ckpt_dir=ckpt_dir, tokenizer_path=tokenizer_path,
    #                         max_seq_len=max_seq_len, max_batch_size=max_batch_size)
    # get_multiple_gt_objects_from_chatgpt(vocab, 5)
    # tokenizer, llama2 = load_llama2_hf()
    gen_data = []
    task_id = int(task_id)
    # for task_id, task_name in tasks.items():

    ###### postive example ########

    if task_id in [3, 4]:
        num_pos_data_per_task = int(
            (1 - neg_ratio - partial_pos_ratio) * num_data_per_task
        )
    else:
        num_pos_data_per_task = int((1 - neg_ratio) * num_data_per_task)
    is_creative_task = True
    for _ in range(num_pos_data_per_task):
        task_name = random.sample(templates[task_id], 1)[0].strip()
        candidate_objects = []
        for object_mask in object_masks:
            if object_mask in task_name:
                is_creative_task = False
                sampled_object = random.sample(vocab, 1)[0]
                task_name = task_name.replace(
                    object_mask, sampled_object.replace("_", " ")
                )
                candidate_objects.append(sampled_object)
        print(task_name)

        # for tasks like "bring me somthing to drink, we ask gpt for GT"
        if is_creative_task:
            ranked_vocab = rank_vocab_by_keyword(vocab, gt_keyword[task_id])
            irr_vocab = ranked_vocab[:-task_vocab_size]
            with open("candidate_objects/" + str(task_id) + ".txt") as f:
                candidate_objects = ast.literal_eval(f.read())
            target_objects = random.sample(candidate_objects, 1)[0]
            num_task_objects = len(target_objects)
            print(target_objects)
            # if not target_objects:
            #     print("Can't find target objects within query_gt_times, skip...")
            #     continue
            context = []
            actions = []
            context_length = random.randrange(num_task_objects + 1, max_context_length)
            for _ in range(context_length - num_task_objects - 1):
                context.append(
                    get_object_image_lvis(random.sample(irr_vocab, 1)[0], vocab)
                )
            recep_id = get_object_image_lvis(predefined_receps[task_id], vocab)
            context.append(recep_id)
            for obj in target_objects:
                context_id = get_object_image_lvis(obj, vocab)
                context.append(context_id)
                actions += wrap_to_plan(context_id, recep_id)
            random.shuffle(context)
            # print({"task": task_name, "context": " ".join(
            #     context), "plan": "###".join(actions)})
            gen_data.append(
                {
                    "task": task_name,
                    "context": " ".join(context),
                    "plan": "###".join(actions),
                }
            )
            with open(
                save_dir + "web_scale_data_generation_" + timestamp + ".json", "w"
            ) as f:
                json.dump(gen_data, f, indent=4)

        # for tasks that asking for specific objects like ovmm-style
        else:
            context = []
            actions = []
            context_length = random.randrange(
                len(candidate_objects), max_context_length
            )

            for _ in range(context_length - len(candidate_objects)):
                context.append(get_object_image_lvis(random.sample(vocab, 1)[0], vocab))

            target_objects_images = []
            for obj in candidate_objects:
                sampled_object_image = get_object_image_lvis(obj, vocab)
                target_objects_images.append(sampled_object_image)
                context.append(sampled_object_image)

            actions = wrap_to_plan_ovmm(task_id, target_objects_images)
            random.shuffle(context)
            gen_data.append(
                {
                    "task": task_name,
                    "context": " ".join(context),
                    "plan": "###".join(actions),
                }
            )
            with open(
                save_dir + "web_scale_data_generation_" + timestamp + ".json", "w"
            ) as f:
                json.dump(gen_data, f, indent=4)

    ###### negative example ########

    is_creative_task = True
    for _ in range(int(neg_ratio * num_data_per_task)):
        task_name = random.sample(templates[task_id], 1)[0].strip()
        candidate_objects = []
        for object_mask in object_masks:
            if object_mask in task_name:
                is_creative_task = False
                sampled_object = random.sample(vocab, 1)[0]
                task_name = task_name.replace(
                    object_mask, sampled_object.replace("_", " ")
                )
                candidate_objects.append(sampled_object)
        print(task_name)

        # for tasks like "bring me somthing to drink, we ask gpt for GT"
        if is_creative_task:
            ranked_vocab = rank_vocab_by_keyword(vocab, gt_keyword[task_id])

            irr_vocab = ranked_vocab[:-task_vocab_size]

            context = []
            actions = []
            context_length = random.randrange(max_context_length)
            for _ in range(context_length):
                context.append(
                    get_object_image_lvis(random.sample(irr_vocab, 1)[0], vocab)
                )

            random.shuffle(context)

            gen_data.append(
                {
                    "task": task_name,
                    "context": " ".join(context),
                    "plan": "###".join(["explore"]),
                }
            )
            with open(
                save_dir + "web_scale_data_generation_" + timestamp + ".json", "w"
            ) as f:
                json.dump(gen_data, f, indent=4)

        # for tasks that asking for specific objects like ovmm-style
        else:
            context = []
            actions = []
            context_length = random.randrange(max_context_length)

            for _ in range(context_length):
                context.append(get_object_image_lvis(random.sample(vocab, 1)[0], vocab))
            filtered_context = []
            for each_context in context:
                need_remove = False
                for obj in candidate_objects:
                    if obj in each_context:
                        need_remove = True
                if not need_remove:
                    filtered_context.append(each_context)

            random.shuffle(filtered_context)
            gen_data.append(
                {
                    "task": task_name,
                    "context": " ".join(filtered_context),
                    "plan": "###".join(["explore"]),
                }
            )
            with open(
                save_dir + "web_scale_data_generation_" + timestamp + ".json", "w"
            ) as f:
                json.dump(gen_data, f, indent=4)

    ###### partial pos (for ovmm) example ########

    if task_id in [3, 4]:
        is_creative_task = True
        for _ in range(int(partial_pos_ratio * num_data_per_task)):
            task_name = random.sample(templates[task_id], 1)[0].strip()
            candidate_objects = []
            for object_mask in object_masks:
                if object_mask in task_name:
                    is_creative_task = False
                    sampled_object = random.sample(vocab, 1)[0]
                    task_name = task_name.replace(
                        object_mask, sampled_object.replace("_", " ")
                    )
                    candidate_objects.append(sampled_object)
            print(task_name)

            # TODO: for tasks like "bring me somthing to drink, we ask gpt for GT"
            # for tasks that asking for specific objects like ovmm-style
            if not is_creative_task:
                context = []
                actions = []
                context_length = random.randrange(
                    len(candidate_objects[:-1]), max_context_length
                )

                for _ in range(context_length - len(candidate_objects[:-1])):
                    sampled_context = get_object_image_lvis(
                        random.sample(vocab, 1)[0], vocab
                    )
                    if candidate_objects[-1] not in sampled_context:
                        context.append(sampled_context)

                target_objects_images = []
                for obj in candidate_objects[:-1]:
                    sampled_object_image = get_object_image_lvis(obj, vocab)
                    target_objects_images.append(sampled_object_image)
                    context.append(sampled_object_image)
                actions = wrap_to_plan_ovmm_partial(task_id, target_objects_images)
                random.shuffle(context)
                gen_data.append(
                    {
                        "task": task_name,
                        "context": " ".join(context),
                        "plan": "###".join(actions),
                    }
                )
                with open(
                    save_dir + "web_scale_data_generation_" + timestamp + ".json", "w"
                ) as f:
                    json.dump(gen_data, f, indent=4)


if __name__ == "__main__":
    # fire.Fire(main)
    main()
