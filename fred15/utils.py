import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable


def process_image(img, normalize=False, requires_grad=False):

    img /= 255.0

    if requires_grad:
        return Variable(torch.from_numpy(img), requires_grad=True)

    else:
        return img


def deprocess_image(dep, img_shape):
    dep *= 255
    dep = np.clip(dep, 0, 255).reshape(img_shape)
    return np.uint8(dep)


def prepare_data_fred_at2(seed, test_size=50):

    datasetFaces = []

    for person in range(1, 41):
        temp = []

        for pose in range(1, 11):
            data = plt.imread("data/archive/s" + str(person) + "/" + str(pose) + ".pgm")
            temp.append(data)

        datasetFaces.append(np.array(temp))

    datasetFaces = np.array(datasetFaces)

    data = []
    for person in range(40):
        for pose in range(10):
            data.append([(1 / 255) * datasetFaces[person, pose, :, :], person])

    random.Random(seed).shuffle(data)

    train, test = data[:-test_size], data[-test_size:]

    X = np.expand_dims(train[0][0], axis=0)
    Y = np.expand_dims(np.array(train[0][1]), axis=0)

    X_test = np.expand_dims(test[0][0], axis=0)
    Y_test = np.expand_dims(np.array(test[0][1]), axis=0)

    for i in range(1, len(train)):

        x, y = np.expand_dims(train[i][0], axis=0), np.expand_dims(
            np.array(train[i][1]), axis=0
        )

        X = np.vstack((X, x))
        Y = np.vstack((Y, y))

    for i in range(1, len(test)):

        x, y = np.expand_dims(test[i][0], axis=0), np.expand_dims(
            np.array(test[i][1]), axis=0
        )

        X_test = np.vstack((X_test, x))
        Y_test = np.vstack((Y_test, y))

    return X, Y, X_test, Y_test


def prepare_data_fred_at(seed):

    # Selection des images et creation de la matrice de donnees correspondante
    indices_pers = [
        i for i in range(15)
    ]  # Indices entre 0 et 14 (15 sujets differents)
    indices_post = [
        i for i in range(11)
    ]  # Indices entre 0 et 10 (11 postures possibles)

    random.Random(seed).shuffle(indices_pers)

    test_post = indices_post
    indices_post = indices_post

    nb_pers = len(indices_pers)
    nb_post = 9

    people_poses = {person: [i for i in range(11)] for person in range(nb_pers)}

    for i, k in enumerate(people_poses, 0):
        random.Random(i).shuffle(people_poses[k])

    people_poses_test = {}

    for k in people_poses:
        people_poses_test[k] = people_poses[k][-2:][:]
        people_poses[k] = people_poses[k][:-2]

    m = nb_pers * nb_post  # Nombre d'images selectionnees pour traitement

    # Creation de la structure de donnees
    n = 243 * 320  # Taille d'une image

    X = np.zeros((m, n))
    y = np.zeros((m, 1))

    # Les differents individus et postures dans le jeu de donnees
    noms_pers = [
        "subject01",
        "subject02",
        "subject03",
        "subject04",
        "subject05",
        "subject06",
        "subject07",
        "subject08",
        "subject09",
        "subject10",
        "subject11",
        "subject12",
        "subject13",
        "subject14",
        "subject15",
    ]
    noms_post = [
        "centerlight",
        "glasses",
        "happy",
        "leftlight",
        "noglasses",
        "normal",
        "rightlight",
        "sad",
        "sleepy",
        "surprised",
        "wink",
    ]

    for i in range(m):
        # Chemin vers l'image (a modifier si necessaire, par exemple sous Google Colab)
        name_im = (
            "data/at&t/"
            + noms_pers[indices_pers[i // nb_post]]
            + "."
            + noms_post[people_poses[indices_pers[i // nb_post]][i % nb_post]]
        )
        # Cette fonction retourne un tableau NumPy contenant l'image sous forme matricielle
        img_array = plt.imread(name_im, format="gif")
        # On stocke l'image sous la forme d'un vecteur ligne
        X[i, :] = np.reshape(img_array, n)
        X[i, :] = (1 / 255) * X[
            i, :
        ]  # Normalisation pour eviter d'avoir de trop gros coefficients
        y[i, :] = people_poses[indices_pers[i // nb_post]][i % nb_post]

    m = nb_pers * 2  # Nombre d'images selectionnees pour traitement
    nb_post = 2
    # Creation de la structure de donnees
    n = 243 * 320  # Taille d'une image

    X_test = np.zeros((m, n))
    y_test = np.zeros((m, 1))

    for i in range(m):
        # Chemin vers l'image (a modifier si necessaire, par exemple sous Google Colab)
        name_im = (
            "data/at&t/"
            + noms_pers[indices_pers[i // nb_post]]
            + "."
            + noms_post[people_poses_test[indices_pers[i // nb_post]][i % nb_post]]
        )
        y_test[i, :] = people_poses_test[indices_pers[i // nb_post]][i % nb_post]
        # Cette fonction retourne un tableau NumPy contenant l'image sous forme matricielle
        img_array = plt.imread(name_im, format="gif")
        # On stocke l'image sous la forme d'un vecteur ligne
        X_test[i, :] = np.reshape(img_array, n)
        X_test[i, :] = (1 / 255) * X_test[
            i, :
        ]  # Normalisation pour eviter d'avoir de trop gros coefficients

    return X, y, X_test, y_test


if __name__ == "__main__":

    prepare_data_fred_at2(0, 50)
