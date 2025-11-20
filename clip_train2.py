#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import math
from typing import List, Dict

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import CLIPModel, CLIPProcessor
from torch.nn.functional import cosine_similarity


# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# In[3]:


BASE_DIR = os.getcwd()
print("BASE_DIR:", BASE_DIR)


# In[4]:


class LoRALinear(nn.Module):
    """
    LoRA wrapper around a Linear layer:
      y = x W^T + (alpha/r) * B(A(x))
    where A: in -> r, B: r -> out.
    """
    def __init__(self, base_layer: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Original (frozen) weight & bias
        self.weight = base_layer.weight
        self.bias = base_layer.bias

        # LoRA trainable weights
        self.lora_A = nn.Linear(self.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, self.out_features, bias=False)

        # Init LoRA
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # Freeze original
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        lora_out = self.lora_B(self.lora_A(x)) * self.scaling
        return base + lora_out


# In[5]:


def apply_lora_to_clip_attn(model: nn.Module, r: int = 8, alpha: float = 16.0):
    """
    Replace all q_proj and v_proj Linear layers in CLIP with LoRALinear.
    """
    for module_name, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child_name in ["q_proj", "v_proj"]:
                lora_layer = LoRALinear(child, r=r, alpha=alpha)
                setattr(module, child_name, lora_layer)


# In[6]:


# Function to generate the correct path for files

def ap(rel_path: str) -> str:
    return os.path.join(BASE_DIR, rel_path)


# In[7]:


# set organization of data

set1 = ["zz/kid-eat-burger"]
set2 = [
    "zz/kids",
    "zz/burgers",
    "zz/kid-and-burger"
]

set3 = [
    "zz/kid-eating-others",
    "zz/others-eat-burger",
    "zz/others"
    ]


# In[8]:


# building dataset based on organization

def data_build(set1, set2, set3):
    relationsets = []

    # ------------------------------------------------------
    # Load ALL text files once and store their lines
    # ------------------------------------------------------
    text_cache = {}   # maps: folder \u2192 list of lines

    def load_text(folder_label):
        if folder_label not in text_cache:
            textfile = ap(folder_label.replace("zz", "text") + ".txt")
            with open(textfile, "r") as f:
                text_cache[folder_label] = [line.strip() for line in f.readlines()]
        return text_cache[folder_label]

    # ------------------------------------------------------
    # Main loop for building relations
    # ------------------------------------------------------
    for i in range(50):
        for j in range(50):

            # pick which set2 / set3 folder to use
            if j < 20:
                _j = 0
            elif j < 40:
                _j = 1
            else:
                _j = 2

            # ------------------------------------------------------
            # Build image file paths
            # ------------------------------------------------------
            im1_index = i + 1          # 1 or 2
            im2_index = (j % 20) + 1   # 1\u201320
            im3_index1 = im2_index
            im3_index2 = ((j + 1) % 20) + 1

            im1file = ap(set1[0].replace("zz", "Data") + f"/1 ({im1_index}).png")
            im2file = ap(set2[_j].replace("zz", "Data") + f"/1 ({im2_index}).png")
            im3file1 = ap(set3[_j].replace("zz", "Data") + f"/1 ({im3_index1}).png")
            try:
                im3file2 = ap(set3[_j].replace("zz", "Data") + f"/1 ({im3_index2}).png")
            except:
                im3_index2 = im3_index2%10
                im3file2 = ap(set3[_j].replace("zz", "Data") + f"/1 ({im3_index2}).png")

            # ------------------------------------------------------
            # Fetch text lines corresponding to each image
            # ------------------------------------------------------
            t1_lines = load_text(set1[0])
            t2_lines = load_text(set2[_j])
            t3_lines = load_text(set3[_j])

            t1 = t1_lines[im1_index - 1]
            t2 = t2_lines[im2_index - 1]
            t3_1 = t3_lines[im3_index1 - 1]
            try:
                t3_2 = t3_lines[im3_index2 - 1]
            except:
                im3_index2 = im3_index2%10
                t3_2 = t3_lines[im3_index2 - 1]

            # ------------------------------------------------------
            # Build relation dictionaries
            # ------------------------------------------------------
            dict1 = {
                "images": [im1file, im2file, im3file1],
                "texts": [t1, t2, t3_1]
            }

            dict2 = {
                "images": [im1file, im2file, im3file2],
                "texts": [t1, t2, t3_2]
            }

            relationsets.append(dict1)
            relationsets.append(dict2)

    return relationsets


# In[9]:


class TripleDataset(Dataset):
    def __init__(self, data: List[Dict]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        imgs = []
        for p in entry["images"]:
            if not os.path.exists(p):
                iid = p[-7]
                if iid == '2':
                    iid = '1'
                else:
                    iid = ''
                ns = p[:-7] + iid + p[-6:]
                p = ns
                if not os.path.exists(p):
                    raise FileNotFoundError(f"Image not found: {p}")
            imgs.append(Image.open(p).convert("RGB"))
        return {"images": imgs, "texts": entry["texts"]}


# In[10]:


def collate_fn(batch: List[Dict]) -> Dict:
    all_images, all_texts = [], []
    for item in batch:
        all_images.extend(item["images"])
        all_texts.extend(item["texts"])
    return {"images": all_images, "texts": all_texts}


# In[11]:


def load_clip_with_manual_lora():
    # 1) Load CLIP on CPU first
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir = "./clip_cache")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir = "./clip_cache")

    # 2) Freeze all original params
    for p in model.parameters():
        p.requires_grad = False

    # 3) Inject LoRA into q_proj and v_proj (still on CPU)
    apply_lora_to_clip_attn(model, r=8, alpha=16.0)

    # 4) Move entire model (including LoRA layers) to device
    model.to(device)

    # 5) Count params
    total, trainable = 0, 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"Total params: {total}, trainable (LoRA): {trainable}")

    return model, processor


# In[12]:


# creating data loader

data = data_build(set1, set2, set3)
dataset = TripleDataset(data)
loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=collate_fn
)
print(len(dataset))


# In[13]:


model, processor = load_clip_with_manual_lora()


# In[14]:


optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr = 1e-3,
)


# In[15]:


model.train()


# In[16]:


# Our customized loss function
def calculate_loss(image_embeds, text_embeds):
    l1 = cosine_similarity(image_embeds[0], text_embeds[0], dim = 0)
    l2 = cosine_similarity(image_embeds[1], text_embeds[1], dim = 0)
    l3 = cosine_similarity(image_embeds[2], text_embeds[2], dim = 0)

    loss = l1 - 0.5*l2 - 0.5*l3

    return loss


# In[17]:


batch = None
for b in loader:
    batch = b
    break

batch


# In[18]:


#start the training here

# from tqdm import tqdm

# num_epochs = 5
# for epoch in range(num_epochs):
#     epoch_loss = 0.0
#     num_batches = 0
#     for batch in tqdm(loader):

#         # Process inputs
#         inputs = processor(
#             text=batch["texts"],
#             images=batch["images"],
#             padding=True,
#             return_tensors="pt",
#         ).to(device)

#         # Forward pass
#         outputs = model(
#             input_ids=inputs["input_ids"],
#             attention_mask=inputs["attention_mask"],
#             pixel_values=inputs["pixel_values"],
#         )

#         image_embeds = outputs.image_embeds
#         text_embeds = outputs.text_embeds

#         # Compute loss
#         loss = calculate_loss(image_embeds, text_embeds)

#         # Backprop
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # Track loss for the epoch
#         epoch_loss += loss.item()
#         num_batches += 1

#     # Average epoch loss
#     avg_loss = epoch_loss / num_batches
#     print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")


# In[19]:


from tqdm import tqdm

num_epochs = 5
for epoch in range(num_epochs):
    epoch_loss = 0.0
    num_batches = 0

    for batch in tqdm(loader):

        image_embeds_list = []
        text_embeds_list = []

        # ---- NEW: Process each pair individually ---- #
        for img, txt in zip(batch["images"], batch["texts"]):

            inputs = processor(
                text=[txt],            # each as a list of length 1
                images=[img],
                padding=True,
                return_tensors="pt",
            ).to(device)

            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"],
            )

            # remove batch dimension [1, 512] → [512]
            image_embeds_list.append(outputs.image_embeds.squeeze(0))
            text_embeds_list.append(outputs.text_embeds.squeeze(0))

        # Stack into shape [3, 512]
        image_embeds = torch.stack(image_embeds_list)
        text_embeds = torch.stack(text_embeds_list)
        # ------------------------------------------------ #

        # Compute loss
        loss = calculate_loss(image_embeds, text_embeds)
        # print(type(loss))

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track loss
        epoch_loss += loss.item()
        num_batches += 1

    # Average epoch loss
    avg_loss = epoch_loss / num_batches
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")


# In[20]:


# model.save_pretrained("./clip_finetuned(1,0.3,0.7)")
model.save_pretrained("./clip_finetuned_demo")


# In[ ]:





# In[21]:


processor.save_pretrained("./clip_finetuned_demo")

