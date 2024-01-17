"""
utilities for performing adversarial attacks on models
"""
import torch

# attacks
def generate_attack(attack_type, model, loss, images, labels, eps):
    if attack_type == "FGSM":
        attack_images = fgsm_attack(model, loss, images, labels, eps)
    else:
        raise NotImplementedError

    return attack_images

def fgsm_attack(model, loss, images, labels, eps, img_range=[0, 255]):
    outputs = model(images)
    
    model.zero_grad()
    cost = loss(outputs, labels)
    cost.backward()

    attack_images = images + eps*images.grad.sign()
    attack_images = torch.clamp(attack_images, img_range[0], img_range[1])

    return attack_images
