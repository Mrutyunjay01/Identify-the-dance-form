from matplotlib import pyplot as plt
import cv2


# Display two images
def display(a, b, title1="Original", title2="Edited"):
    plt.subplot(121), plt.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(b), plt.title(title2)
    plt.xticks([]), plt.yticks([])
    plt.show()


# Display one image
def display_one(a, title1="Original"):
    plt.imshow(a), plt.title(title1)
    plt.show()


def preprocessing(imageset):
    # read 3 images to work
    resultant_images = []
    for i in range(len(imageset)):

        img = imageset[i]

        try:
            print('Original size : ', img.shape)
        except AttributeError:
            print('Shape not found.')

        # set up the target shape
        width = 440
        height = 440
        dim = (width, height)
        resultantImage = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)

        # sanity check
        try:
            print('Resized shape: ', resultantImage.shape)
        except AttributeError:
            print('Not preprocessed')
        # normalize the resultant image
        # resultantImage /= 255.0
        # visualize both
        # display(img, resultantImage)

        resultant_images.append(resultantImage)

    return resultant_images
