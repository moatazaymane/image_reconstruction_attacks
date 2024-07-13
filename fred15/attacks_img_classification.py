import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from tqdm import tqdm
from utils import deprocess_image, process_image


def attack_classification(
    model, target_label, img_shape, iterations, model_path, learning_rate=0.2
):

    img_features = img_shape[0] * img_shape[1]
    state = torch.load(model_path)
    model.load_state_dict(state["model_state_dict"])

    noise_image = np.uint8(np.random.uniform(0, 255, (1, img_features)))

    variable = process_image(
        noise_image.astype(np.float32), normalize=False, requires_grad=True
    )

    optimizer = optim.Adam(
        [variable], lr=learning_rate, weight_decay=0.01
    )

    for iteration in tqdm(range(iterations), desc="computing"):

        # running inference and getting the target label logit | we run gradient ascent to maximize the class logit
        target_logit = -model(variable)[:, target_label]

        model.zero_grad()

        target_logit.backward()

        optimizer.step()

        if iteration % 100 == 0:
            reconstructed_image = deprocess_image(
                torch.clone(variable).detach().numpy(), img_shape=img_shape
            )
            plt.imsave(
                f"reconstructed_images/{target_label}_{iteration}.png",
                reconstructed_image,
                cmap="gray",
            )


if __name__ == "__main__":

    from code.fred15.models import Softmax_Model

    from config import atet_classes, atet_im_shape

    model = Softmax_Model(
        width=atet_im_shape[0] * atet_im_shape[1], out_features=atet_classes
    )
    attack_classification(
        model=model,
        model_path="models/softmax_model",
        target_label=7,
        img_shape=atet_im_shape,
        iterations=205,
        learning_rate=0.2,
    )
