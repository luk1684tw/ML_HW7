import os
from datetime import datetime as dt

import numpy as np

from PIL import Image
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist


train_path = os.path.join('Yale_Face_Database', 'Training')
test_path = os.path.join('Yale_Face_Database', 'Testing')


def linear_kernel(x, y):
    k = list()
    for img in x:
        k.append(np.dot(img, y.T))
    k = np.array(k)

    return k


def rbf_kernel(x, y):
    k = cdist(x, y, metric='sqeuclidean')
    k = np.exp(-gamma*k)

    return k


def fetch_eigvecs(eig_val, eig_vec, k, kernel=None):
    eig_dict = dict(zip(eig_val, eig_vec.T))
    eig_val[::-1].sort()
    principle_component = eig_dict[eig_val[0]]
    for i in range(1, k):
        if kernel:
            principle_component = np.vstack((principle_component, eig_dict[eig_val[i]] / eig_val[i]))
        else:
            principle_component = np.vstack((principle_component, eig_dict[eig_val[i]]))

    return principle_component.T


def KPCA(X, test, res_dim, kernel, n_neighbors):
    # X row-based
    k = kernel_funcs[kernel](X, X)
    print ('kerenl shape:', k.shape)
    
    N = len(X)
    ones = np.ones((N, N)) / N
    k = k - np.dot(ones, k) - np.dot(k, ones) + np.dot(ones, np.dot(k, ones))

    eig_val, eig_vec = np.linalg.eig(k)
    alpha = fetch_eigvecs(eig_val, eig_vec, res_dim, kernel=1) # 135X25
    print (alpha.shape)
    train_img_lowD = np.dot(k, alpha)

    prd_result = list()
    for img in test:
        img = img.reshape(1, -1)
        
        img_lowD = kernel_funcs[kernel](X, img).flatten()
        img_lowD = np.sum((alpha.T*img_lowD).T, axis=0)
        
        distance = list()
        for img in train_img_lowD:
            distance.append(np.linalg.norm(img - img_lowD))
        distance = np.array(distance)
        # print (distance)
        idx = np.argpartition(distance, n_neighbors)[:n_neighbors] // 9 + 1
        prd_result.append(np.argmax(np.bincount(idx)))

    return np.array(prd_result)


def PCA(X, res_dim):
    print ('Find PC from input array')
    (m, dim) = X.shape

    avg_face = np.mean(X, axis=0)
    X = X - np.tile(avg_face, (m, 1))
    cov = np.dot(X, X.T)
    eig_val, eig_vec = np.linalg.eig(cov)
    eig_vec /= np.linalg.norm(eig_vec, axis=0)

    principle_component = fetch_eigvecs(eig_val, eig_vec.real, res_dim)
    print (principle_component.shape)
    
    Y = np.dot(X.T, principle_component).astype('float32')
    Y /= np.linalg.norm(Y, axis=0)
    print (Y.shape)

    return Y, avg_face


def predict_PCA(train_data, test_data, avg_face, train_W, n_neighbors):
    prd_result = list()
    diff_train = train_data - avg_face
    diff_train /= np.linalg.norm(diff_train, axis=1)[:, None]

    for image in test_data:
        diff = image - avg_face
        diff /= np.linalg.norm(diff)

        img_lowD = np.dot(train_W.T, diff)
        dist = list()
        for train_img in diff_train:
            train_lowD = np.dot(train_W.T, train_img)
            dist.append(np.linalg.norm(train_lowD - img_lowD))
        dist = np.array(dist)
        idx = np.argpartition(dist, n_neighbors)[:n_neighbors] // 9 + 1
        prd_result.append(np.argmax(np.bincount(idx)))

    return np.array(prd_result)


def KLDA(train_imgs, k, kernel):
    print (train_imgs.shape) # should be 135 X 10000
    start = dt.now()
    center_set = list()
    within_scatter, between_scatter = 0, 0
    print ('Start calculate SW...')
    for i in range(15):
        # calculate Sk and sum up all Sks
        print (i)
        data_i = train_imgs[9 * i : 9 * (i + 1), :]
        center = np.mean(data_i, axis=0)
        center_set.append(center)

        for data in data_i:
            diff = (data - center).reshape((-1, 1)).astype('float32')
            within_scatter += kernel_funcs[kernel](diff, diff)

    print (np.linalg.det(within_scatter))
    center_set = np.array(center_set)
    print (center_set.shape)
    print ('Start calculate SB...')
    center = np.mean(train_imgs, axis=0)
    for center_i in center_set:
        diff = (center_i - center).reshape((-1, 1)).astype('float32')
        between_scatter += 9 * kernel_funcs[kernel](diff, diff)

    print ((dt.now() - start).total_seconds())
    print ('Start calculate W...')
    eig_val, eig_vec = np.linalg.eig((np.linalg.pinv(within_scatter) * between_scatter))
    W = fetch_eigvecs(eig_val, eig_vec.real, k)
    # plot_face(5, 5, W.T, 'KLDA_FisherFace.png', (100, 100))

    return W



def LDA(train_imgs, k):
    print (train_imgs.shape)
    start = dt.now()
    center_set = list()
    within_scatter, between_scatter = 0, 0
    print ('Start calculate SW...')
    for i in range(15):
        # calculate Sk and sum up all Sks
        data_i = train_imgs[9 * i : 9 * (i + 1), :]
        center = np.mean(data_i, axis=0)
        center_set.append(center)

        for data in data_i:
            diff = (data - center).reshape((-1, 1)).astype('float32')
            within_scatter += np.dot(diff, diff.T)

    print (np.linalg.det(within_scatter))
    center_set = np.array(center_set)
    print (center_set.shape)
    print ('Start calculate SB...')
    center = np.mean(train_imgs, axis=0)
    for center_i in center_set:
        diff = (center_i - center).reshape((-1, 1)).astype('float32')
        between_scatter += 9 * np.dot(diff, diff.T)

    print ((dt.now() - start).total_seconds())
    print ('Start calculate W...')
    eig_val, eig_vec = np.linalg.eig((np.linalg.pinv(within_scatter) * between_scatter))
    W = fetch_eigvecs(eig_val, eig_vec.real, k)
    plot_face(5, 5, W.T, 'LDA_FisherFace.png', (100, 100))

    return W


def predict_LDA(train_imgs, test_imgs, W, n_neighbors):
    prd_result = list()
    print ('W shape', W.shape)
    # print ('train shape', train_imgs_lowD.shape)

    for img in test_imgs:
        img_lowD = np.dot(img, W)

        distance = list()
        for train_img in train_imgs:
            train_img_lowD = np.dot(train_img, W)
            distance.append(np.linalg.norm(train_img_lowD - img_lowD))
        distance = np.array(distance)
        idx = np.argpartition(distance, n_neighbors)[:n_neighbors] // 9 + 1
        prd_result.append(np.argmax(np.bincount(idx)))

    return np.array(prd_result)


def plot_face(axis_x, axis_y, eigen_face, img_name, scale):
    # eigen_face dimension of (K components X scale)
    eig_face, row = list(), list()
    for i in range(axis_y):
        for j in range(axis_x):
            eig_face.append(eigen_face[i * axis_x + j, :].reshape(scale))
        row.append(np.hstack(eig_face)) # extend 3D 5 eigenfaces to 2D array
        eig_face = list()
    eig_faces = np.vstack(row) #extend 3D to 2D

    plt.imshow(eig_faces, cmap='gray')
    plt.imsave(img_name, eig_faces, cmap='gray')

    return


def read_images(path, scale=(195, 231)):
    dataset = list()
    for image in os.listdir(path):
        with Image.open(os.path.join(path, image)) as img:
            img = img.resize(scale, Image.ANTIALIAS)
            dataset.append(np.array(img).flatten())
    
    # returns a np array with shape (num_imgs)X(num_pixels_per_image)
    return np.array(dataset)


if __name__ == "__main__":
    # Global hyper-parameters
    k = 25
    reconst = np.random.randint(135, size=10)
    gamma = 5e-9 # used in rbf kernel
    kernel_funcs = {'linear': linear_kernel, 'rbf': rbf_kernel}

    # task1 & task2 of PCA
    train_imgs = read_images(train_path, (195, 231)) # note PIL scale order
    print (train_imgs.shape)
    train_W, avg = PCA(train_imgs, k)
    print ('eigen face shape:' , train_W.shape)
    print ('PCA done')
    plot_face(5, 5, train_W.T, 'PCA_EigenFace.png', (231, 195))

    print ('Reconstructing random 10 images')
    print (reconst)
    plot_face(5, 2, train_imgs[reconst], 'OriginFace.png', (231, 195))
    res_mean = np.mean(train_imgs[reconst], axis=0)
    plot_face(5, 2, np.dot(train_imgs[reconst] - avg, np.dot(train_W, train_W.T)) + avg, 'PCA_Resconstruct.png', (231, 195))

    print ('Predicting test images')
    test_imgs = read_images(test_path)
    test_labels = sorted([i for i in range(1, 16)]*2)

    prd_result = predict_PCA(train_imgs, test_imgs, avg, train_W, 3)
    print ('Accuracy of PCA: ', len(prd_result[prd_result == test_labels]) / 30)

    # kernel PCA
    prd_linear = KPCA(train_imgs, test_imgs, k, 'linear', 3)
    prd_rbf = KPCA(train_imgs, test_imgs, k, 'rbf', 3)
    print (prd_linear)
    print ('Accuracy of KPCA_rbf: ', len(prd_rbf[prd_rbf == test_labels]) / 30)
    print ('Accuracy of KPCA_linear: ', len(prd_linear[prd_linear == test_labels]) / 30)

    # task1 & task2 of LDA
    print ('Start LDA')
    start = dt.now()
    train_imgs = read_images(train_path, (100, 100))
    test_imgs = read_images(test_path, (100, 100))
    W = LDA(train_imgs, k)
    print ((dt.now() - start).total_seconds())

    print ('Reconstructing random 10 images')
    plot_face(5, 2, np.dot(np.dot(train_imgs[reconst], W), W.T), 'LDA_Reconstruct.png', (100, 100))


    print ('Predicting test images')
    prd_result = predict_LDA(train_imgs,test_imgs, W, 3)
    print (prd_result)
    print ('Accuracy of LDA: ', len(prd_result[prd_result == test_labels]) / 30)

    # kernel FLD
    W = KLDA(train_imgs, k, 'rbf')
    prd_result = predict_LDA(train_imgs,test_imgs, W, 3)
    print (prd_result)
    print ('Accuracy of RBF KLDA: ', len(prd_result[prd_result == test_labels]) / 30)

    W = KLDA(train_imgs, k, 'linear')
    prd_result = predict_LDA(train_imgs,test_imgs, W, 3)
    print (prd_result)
    print ('Accuracy of Linear KLDA: ', len(prd_result[prd_result == test_labels]) / 30)