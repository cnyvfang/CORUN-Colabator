import statistics

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.autograd import Variable

from torchvision import transforms

import PIL.Image as Image
import pickle
import os


class Memory_bank(nn.Module):
    def __init__(self):
        super(Memory_bank, self).__init__()

        self.dict={
        }

    def get(self, image_name):
        if image_name in self.dict:
            return self.dict[image_name]
        else:
            return None

    def update(self, image_name, image, transmission, musiq, clip_score, mask=None):
        if mask is not None:
            self.dict.update({image_name: [image, transmission, musiq, clip_score, mask]})
        else:
            self.dict.update({image_name: [image, transmission, musiq, clip_score, None]})

    def save(self, path, current_iter):
        # mkdir path/current_iter
        path = os.path.join(path, "memory_bank", str(current_iter) + '/')
        if not os.path.exists(path):
            os.makedirs(path)

        # save dict to file
        with open(path + 'memory_bank.pkl', 'wb') as f:
            pickle.dump(self.dict, f)

        # save all images in memory bank
        for key in self.dict.keys():
            image = self.dict[key][0]
            transforms.ToPILImage()(image).save(path + key)

    def load(self, path):
        # load dict from file
        with open(path, 'rb') as f:
            self.dict = pickle.load(f)

    @torch.no_grad()
    def forward(self, image_name_list, image_list, transmission_list, musiq_list, clip_score_list, device, mask=None):
        pseudo_label = []
        pseudo_transmission = []
        teacher_musiq = []
        teacher_clip_score = []
        mask_stack = []
        for i in range(len(image_name_list)):
            temp_image_data = self.get(image_name_list[i])
            # init
            if temp_image_data is None:
                pseudo_label.append(image_list[i])
                pseudo_transmission.append(transmission_list[i])
                teacher_musiq.append(musiq_list[i])
                teacher_clip_score.append(clip_score_list[i])
                if mask is not None:
                    mask_stack.append(mask[i])
                    self.update(image_name_list[i], image_list[i].detach().cpu().clone(), transmission_list[i].detach().cpu().clone(), musiq_list[i].detach().cpu().clone(), clip_score_list[i].detach().cpu().clone(), mask[i].detach().cpu().clone())
                else:
                    self.update(image_name_list[i], image_list[i].detach().cpu().clone(), transmission_list[i].detach().cpu().clone(), musiq_list[i].detach().cpu().clone(), clip_score_list[i].detach().cpu().clone())
            # update
            else:
                temp_musiq = musiq_list[i].detach().cpu()
                temp_clip_score = clip_score_list[i].detach().cpu()
                if temp_image_data[2] <= temp_musiq and temp_image_data[3] <= temp_clip_score:
                    pseudo_label.append(image_list[i].to(device))
                    pseudo_transmission.append(transmission_list[i].to(device))
                    teacher_musiq.append(musiq_list[i].to(device))
                    teacher_clip_score.append(clip_score_list[i].to(device))
                    if mask is not None:
                        mask_stack.append(mask[i].to(device))
                        self.update(image_name_list[i], image_list[i].detach().cpu().clone(), transmission_list[i].detach().cpu().clone(), temp_musiq.clone(), temp_clip_score.clone(), mask[i].detach().cpu().clone())
                    else:
                        self.update(image_name_list[i], image_list[i].detach().cpu().clone(), transmission_list[i].detach().cpu().clone(), temp_musiq.clone(), temp_clip_score.clone())
                else:
                    pseudo_label.append(temp_image_data[0].to(device))
                    pseudo_transmission.append(temp_image_data[1].to(device))
                    teacher_musiq.append(temp_image_data[2].to(device))
                    teacher_clip_score.append(temp_image_data[3].to(device))
                    if mask is not None:
                        mask_stack.append(temp_image_data[4].to(device))

        if mask is not None:
            return torch.stack(pseudo_label).detach(), torch.stack(pseudo_transmission).detach(), torch.stack(teacher_musiq).detach(), torch.stack(teacher_clip_score).detach(), torch.stack(mask_stack).detach()
        else:
            return torch.stack(pseudo_label).detach(), torch.stack(pseudo_transmission).detach(), torch.stack(teacher_musiq).detach(), torch.stack(teacher_clip_score).detach()


class Memory_bank_woT(nn.Module):
    def __init__(self):
        super(Memory_bank_woT, self).__init__()

        self.dict={
        }

    def get(self, image_name):
        if image_name in self.dict:
            return self.dict[image_name]
        else:
            return None

    def update(self, image_name, image, musiq, clip_score, mask=None):
        if mask is not None:
            self.dict.update({image_name: [image, musiq, clip_score, mask]})
        else:
            self.dict.update({image_name: [image, musiq, clip_score, None]})

    def save(self, path, current_iter):
        # mkdir path/current_iter
        path = os.path.join(path, "memory_bank", str(current_iter) + '/')
        if not os.path.exists(path):
            os.makedirs(path)

        # save dict to file
        with open(path + 'memory_bank.pkl', 'wb') as f:
            pickle.dump(self.dict, f)

        # save all images in memory bank
        for key in self.dict.keys():
            image = self.dict[key][0]
            transforms.ToPILImage()(image).save(path + key)

    def load(self, path):
        # load dict from file
        with open(path, 'rb') as f:
            self.dict = pickle.load(f)

    @torch.no_grad()
    def forward(self, image_name_list, image_list, musiq_list, clip_score_list, device, mask=None):
        pseudo_label = []
        teacher_musiq = []
        teacher_clip_score = []
        mask_stack = []
        for i in range(len(image_name_list)):
            temp_image_data = self.get(image_name_list[i])
            if temp_image_data is None:
                pseudo_label.append(image_list[i])
                teacher_musiq.append(musiq_list[i])
                teacher_clip_score.append(clip_score_list[i])
                if mask is not None:
                    mask_stack.append(mask[i])
                    self.update(image_name_list[i], image_list[i].detach().cpu().clone(),  musiq_list[i].detach().cpu().clone(), clip_score_list[i].detach().cpu().clone(), mask[i].detach().cpu().clone())
                else:
                    self.update(image_name_list[i], image_list[i].detach().cpu().clone(),  musiq_list[i].detach().cpu().clone(), clip_score_list[i].detach().cpu().clone())
            else:
                temp_musiq = musiq_list[i].detach().cpu()
                temp_clip_score = clip_score_list[i].detach().cpu()
                if temp_image_data[1] <= temp_musiq and temp_image_data[2] <= temp_clip_score: # better than previous
                    pseudo_label.append(image_list[i].to(device))
                    teacher_musiq.append(musiq_list[i].to(device))
                    teacher_clip_score.append(clip_score_list[i].to(device))
                    if mask is not None:
                        mask_stack.append(mask[i].to(device))
                        self.update(image_name_list[i], image_list[i].detach().cpu().clone(), temp_musiq.clone(), temp_clip_score.clone(), mask[i].detach().cpu().clone())
                    else:
                        self.update(image_name_list[i], image_list[i].detach().cpu().clone(), temp_musiq.clone(), temp_clip_score.clone())
                else:
                    pseudo_label.append(temp_image_data[0].to(device))
                    teacher_musiq.append(temp_image_data[1].to(device))
                    teacher_clip_score.append(temp_image_data[2].to(device))
                    if mask is not None:
                        mask_stack.append(temp_image_data[3].to(device))

        if mask is not None:
            return torch.stack(pseudo_label).detach(), torch.stack(teacher_musiq).detach(), torch.stack(teacher_clip_score).detach(), torch.stack(mask_stack).detach()
        else:
            return torch.stack(pseudo_label).detach(), torch.stack(teacher_musiq).detach(), torch.stack(teacher_clip_score).detach()

