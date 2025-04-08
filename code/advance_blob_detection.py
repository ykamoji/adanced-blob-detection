import numpy as np
import os
from utils import imread
from skimage.color import rgb2gray
from scipy.signal import convolve2d
from skimage.transform import resize
import time
from drawBlobs import drawBlobs


def get_meshgrid(sigma):
    kernel_size = np.round(6 * sigma)
    if kernel_size % 2 == 0:
        kernel_size += 1
    half_size = np.floor(kernel_size / 2)
    return np.meshgrid(np.arange(-half_size, half_size + 1), np.arange(-half_size, half_size + 1))


def laplacian_of_gaussian_filter(sigma):
    x, y = get_meshgrid(sigma)
    sigma_squared = sigma ** 2
    sum_squared = x ** 2 + y ** 2
    term = -(1 / sigma_squared) * (1 - (sum_squared / (2 * sigma_squared)))
    exp_term = np.exp(-sum_squared / (2 * sigma_squared))
    kernel = term * exp_term
    kernel = kernel / np.sum(np.abs(kernel))
    return kernel


def difference_of_gaussian_filter(sigma, k):
    x, y = get_meshgrid(sigma)
    sigma_squared = sigma ** 2
    sum_squared = x ** 2 + y ** 2

    gaussian_1 = np.exp(-sum_squared / (2 * sigma_squared * (k ** 2)))
    gaussian_1 = gaussian_1 / np.sum(np.abs(gaussian_1))

    gaussian_2 = np.exp(-sum_squared / (2 * sigma_squared))
    gaussian_2 = gaussian_2 / np.sum(np.abs(gaussian_2))

    return gaussian_1 - gaussian_2


def max_suppression(array, n):
    rows, cols = array.shape
    res_arr = np.zeros([rows, cols])
    for i in range(rows): 
        for j in range(cols):
            if i - (n - 1) / 2 < 0:
                a = 0
            else:
                a = int(i - (n - 1) / 2)
            if j - (n - 1) / 2 < 0:
                b = 0
            else:
                b = int(j - (n - 1) / 2)
            if i + 1 + (n - 1) / 2 > rows:
                a1 = int(rows)
            else:
                a1 = int(i + 1 + (n - 1) / 2)
            if (j + 1 + (n - 1) / 2) > cols:
                b1 = int(cols)
            else:
                b1 = int(j + 1 + (n - 1) / 2)
            neigh = array[a:a1, b:b1]
            res_arr[i, j] = np.max(neigh)
    return res_arr


def create_scale_space(img, sigma, k, levels, filter):
    original_row, original_col = np.shape(img)
    scale_space = np.zeros([img.shape[0], img.shape[1], levels])
    suppressed_space = np.zeros([img.shape[0], img.shape[1], levels])

    for i in range(levels):
        if filter == 'LOG':
            kernel = laplacian_of_gaussian_filter(sigma)
        else:
            kernel = difference_of_gaussian_filter(sigma, k)
        down_image_row = int(np.floor(original_row/ k**i))
        down_image_col = int(np.floor(original_col/ k**i))
        image_down = resize(img, [down_image_row, down_image_col], anti_aliasing=True)
        convolved_image = convolve2d(image_down, kernel, mode='same')
        convolved_image = np.power(convolved_image, 2)
        scale_space[:, :, i] = resize(convolved_image, [original_row, original_col], anti_aliasing=True)
        suppressed_space[:, :, i] = max_suppression(scale_space[:, :, i], 7)
        img = convolve2d(img, np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) * 1/9, mode='same')
        print(f"\rCompleted {(i + 1) * 100 / levels:.2f} %", end=" ", flush=True)

    return scale_space, suppressed_space


def find_scale(org_row, org_col, suppressed_space):
    max_scale = np.zeros(suppressed_space.shape)
    for i in range(org_row):
        for j in range(org_col):
            ind = np.unravel_index(np.argmax(suppressed_space[i, j, :]), suppressed_space[i, j, :].shape)
            max_scale[i, j, ind[0]] = suppressed_space[i, j, ind[0]]
    return max_scale


def find_blobs(max_scale, threshold, sigma, k):
    coord = []
    for i in range(max_scale.shape[2]):
        ind = np.argwhere(max_scale[:, :, i] >= threshold)
        radius = 2 ** 0.5 * sigma * (k ** i)
        for point in ind:
            if (max_scale.shape[1] - 15) > point[1] > 15 and (max_scale.shape[0] - 15) > point[0] > 15:
                coord.append([point[1], point[0], radius, 100, 1.0])

    return coord


def edge_detector(blobs, scale_space, initial_sigma, k, levels):
    sobelFilter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

    final_blobs = []
    sigma = initial_sigma
    for i in range(levels):
        sigma = sigma * k
        image_grad_x = convolve2d(scale_space[:, :, i], sobelFilter, mode='same')
        image_grad_y = convolve2d(scale_space[:, :, i], sobelFilter.T, mode='same')
        kernel = laplacian_of_gaussian_filter(sigma)
        image_x = convolve2d(np.square(image_grad_x), kernel, mode='same')
        image_y = convolve2d(np.square(image_grad_y), kernel, mode='same')
        image_xy = convolve2d(image_grad_x * image_grad_y, kernel, mode='same')

        det = (image_x * image_y) - (image_xy ** 2)
        trace = image_x + image_y
        R = det - 0.05 * (trace ** 2)
        for blob in blobs:
            if blob[2] == i and R[blob[1], blob[0]] > 0:
                final_blobs.append(blob)

    # print(f"Removed {len(final_blobs) - len(blobs)} blobs from harris edge detector")
    return final_blobs


def detectBlobs(im, params):
    image_gray = rgb2gray(im)
    org_row, org_col = image_gray.shape
    scale_space, suppressed_space = create_scale_space(image_gray, params.initial_sigma, params.k, params.levels, params.filter)
    max_scale = find_scale(org_row, org_col, suppressed_space)
    max_scale = np.multiply(max_scale, (max_scale == scale_space))
    blobs = find_blobs(max_scale, params.threshold, params.initial_sigma, params.k)
    # blobs = edge_detector(blobs, max_scale, params.initial_sigma, params.k ,params.levels)
    return np.array(blobs)


images = ['butterfly.jpg', 'einstein.jpg', 'faces.jpg', 'fishes.jpg', 'football.jpg', 'sunflowers.jpg']

class Params:
    def __init__(self, levels=10, initial_sigma=2, k=2**0.35, threshold=0.0001):
        self.levels = levels
        self.initial_sigma = initial_sigma
        self.k = k
        self.threshold = threshold

    def set_filter_method(self, filter):
        self.filter = filter


## LOG
paramsMap = {
    'butterfly.jpg': Params(),
    'einstein.jpg': Params(),
    'faces.jpg': Params(),
    'fishes.jpg': Params(),
    'football.jpg': Params(),
    'sunflowers.jpg': Params()
}

results = {}
datadir = os.path.join('..', 'data', 'blobs')
for imageName in images:
    imName = imageName.split('.')[0]
    im = imread(os.path.join(datadir, imageName))
    print(f"Detecting blobs for {imName}:")
    results[imName] = {}
    for filter in ['LOG','DOG']:
        start = time.time()
        params = paramsMap[imageName]
        params.set_filter_method(filter)
        blobs = detectBlobs(im, params)
        results[imName][filter] = (params, len(blobs), time.time() - start)
        # print(f"Time taken = {time.time() - start:.4f}")
        drawBlobs(im, blobs, 1000, imName, params, save_path="../output/blob_detector_extra/")


print("\nCompleted !\n")
print("Image"+"\t"*2+"Filter"+"\t"*2+"Parameters"+"\t"*6+"Blobs"+"\t"*2+"Time")
print("-"*120)


def print_params(params):
    return f"level={params.levels}, sigma={params.initial_sigma}, k={params.k:.3f}, threshold={params.threshold}"


for image_name, data in results.items():
    for filter, (params, blob_count, duration) in data.items():
        print(f"{image_name.ljust(15, ' ')}"+ "\t" + f"{filter}" + "\t" * 2 +
              f"{print_params(params)}" + "\t" * 2 + f"{blob_count}" + "\t" * 2 + f"{duration:.3f}")
print("-" * 120)