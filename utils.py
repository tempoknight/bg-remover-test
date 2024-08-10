import cv2

class Utils:
    def __init__(self):
        pass

    
    def splitter(self, image, num_of_row, num_of_col):
        h, w, c = image.shape
        img_list = []
        for i in range(0, h, h//num_of_row):
            for j in range(0, w, w//num_of_col):
                img_list.append(image[i:i+h//num_of_row, j:j+w//num_of_col])
        return img_list


if __name__ == "__main__":
    u = Utils()
    image = cv2.imread("static/example1.jpg")
    images = u.splitter(image, 5, 5 )
    for i,image in enumerate(images):
        cv2.imwrite(f"test/example{i}_split.jpg", image)

