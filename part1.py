import os


import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

train_path = os.path.join('Yale_Face_Database', 'Training')
test_path = os.path.join('Yale_Face_Database', 'Testing')


def PCA(X, res_dim):
    print ('Find PC from input array')
    (m, dim) = X.shape

    avg_face = np.mean(X, axis=0)
    X = X - np.tile(avg_face, (m, 1))
    cov = np.dot(X, X.T)
    eig_val, eig_vec = np.linalg.eig(cov)
    eig_vec /= np.linalg.norm(eig_vec, axis=0)

    # find k largest eigenvectors
    eig_dict = dict(zip(eig_val, eig_vec.T))
    eig_val[::-1].sort()
    principle_component = eig_dict[eig_val[0]]
    for i in range(1, res_dim):
        principle_component = np.vstack((principle_component, eig_dict[eig_val[i]]))
    principle_component = principle_component.T
    print (principle_component.shape)
    
    Y = np.dot(X.T, principle_component).astype('float32')
    Y /= np.linalg.norm(Y, axis=0)
    print (Y.shape)

    return Y, avg_face


def predict_PCA(train_data, test_data, n_neighbors):
    prd_result = list()
    for image in test_data:
        distance = np.linalg.norm(train_data - image, axis=1)
        print (distance.shape)
        neighbors = np.argpartition(distance, -n_neighbors)[-n_neighbors:]# // 9 + 1
        # n_neighbors = distance.argsort()[-n_neighbors:][::-1]
        print (neighbors)
        prd_result.append(np.bincount(neighbors).argmax())

    return np.array(prd_result)


def plot_face(axis_x, axis_y, eigen_face, img_name):
    # eigen_face dimension of (K components X 231*195)
    eig_face, row = list(), list()
    for i in range(axis_y):
        for j in range(axis_x):
            eig_face.append(eigen_face[i * axis_x + j, :].reshape((231, 195)))
        row.append(np.hstack(eig_face)) # extend 3D 5 eigenfaces to 2D array
        eig_face = list()
    eig_faces = np.vstack(row) #extend 3D to 2D

    plt.imshow(eig_faces, cmap='gray')
    plt.imsave(img_name, eig_faces, cmap='gray')

    return


def read_images(path):
    dataset = list()
    for image in os.listdir(path):
        with Image.open(os.path.join(path, image)) as img:
            dataset.append(np.array(img).flatten())
    
    # returns a np array with shape (num_imgs)X(num_pixels_per_image)
    return np.array(dataset)


if __name__ == "__main__":
    # task1 & task2 of PCA
    k = 25
    train_imgs = read_images(train_path)
    print (train_imgs.shape)
    train_W, avg = PCA(train_imgs, k)
    print (avg.shape)
    print ('eigen face shape:' , train_W.shape)
    print ('PCA done')
    plot_face(5, 5, train_W.T, 'PCA_EigenFace.png')

    # print ('Reconstructing random 10 images')
    # reconst = np.random.randint(len(train_imgs), size=10)
    # print (reconst)
    # plot_face(5, 2, train_imgs[reconst], 'PCA_Origin.png')
    # res_mean = np.mean(train_imgs[reconst], axis=0)
    # plot_face(5, 2, np.dot(train_imgs[reconst] - avg, np.dot(train_W, train_W.T)) + avg, 'PCA_Resconstruct.png')

    print ('Predicting test images')
    test_imgs = read_images(test_path)
    test_imgs_lowD = np.dot(test_imgs - avg, train_W) # row-based
    test_imgs_lowD /= np.linalg.norm(test_imgs_lowD, axis=1)[:, None]
    print (test_imgs_lowD.shape)

    train_imgs_lowD = np.dot(train_imgs - avg, train_W) # row-based
    train_imgs_lowD /= np.linalg.norm(train_imgs_lowD, axis=1)[:, None]
    print (train_imgs_lowD.shape)
    
    test_labels = sorted([i for i in range(1, 16)]*2)

    prd_result = predict_PCA(train_imgs_lowD, test_imgs_lowD, 5)
    print (prd_result // 9 + 1)
    # print (len(prd_result[prd_result == test_labels]))
    