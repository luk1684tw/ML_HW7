import os


import numpy as np
from PIL import Image

train_path = os.path.join('Yale_Face_Database', 'Training')
test_path = os.path.join('Yale_Face_Database', 'Testing')


def PCA(X, res_dim):
    print ('Find PC from input array')
    # (dim, m) = X.shape
    # print (dim, m)
    # avg_face = np.mean(X, axis=1)
    # X = X - avg_face.reshape((-1, 1))
    # cov = np.dot(X, X.T) / dim
    # eig_val, eig_vec = np.linalg.eig(cov)
    # eig_vec /= np.sum(eig_vec, axis=0)

    # eig_dict = dict(zip(eig_val, eig_vec))
    # eig_val[::-1].sort()
    # PCs = list()
    # print (eig_val)
    # for i in range(res_dim):
    #     PCs.append(eig_dict[eig_val[i]])

    # eig_face = np.array(PCs).real.T
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))

    return M[:, 0:res_dim]# , avg_face


def read_images(path):
    individual_dict = dict()
    images_name = os.listdir(path) 
    for i in range(15):
        indiv_images = images_name[9 * i : 9*(i+1)]
        indiv_list = list()
        for image in indiv_images:
            with Image.open(os.path.join(path, image)) as img:
                img = np.array(img.resize((resize_pixels, resize_pixels), Image.ANTIALIAS)).flatten()
                indiv_list.append(img)
        individual_dict[i] = np.array(indiv_list).T

    return individual_dict


if __name__ == "__main__":
    # task1 & task2
    k = 25
    resize_pixels = 100
    train_imgs = read_images(train_path)

    for key, val in train_imgs.items():
        print(val.shape)
        train_W = PCA(val.T, k)
        print (train_W.shape)
        print ('PCA done')
        for eigenface in train_W.T:
            print (eigenface)
            img = Image.fromarray((eigenface*255).astype('uint8').reshape(resize_pixels, resize_pixels))
            img.show()