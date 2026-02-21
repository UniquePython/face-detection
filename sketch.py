import numpy as np
import numpy.typing as npt
import cv2


def convert_to_grayscale(img_path: str) -> npt.NDArray[np.uint8]:
	image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
	if image is not None:
		return image
	else:
		raise ValueError("Error opening the image file.")


def invert_img(img: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
	return np.invert(img)


def gaussian_blur(img: npt.NDArray[np.uint8], blur_strength: float = 0.05) -> npt.NDArray[np.uint8]:
	height, width = img.shape
	kernel_width = 2 * int(width * blur_strength) + 1
	kernel_height = 2 * int(height * blur_strength) + 1
	return cv2.GaussianBlur(img, (kernel_width, kernel_height), 0)


def color_dodge_blend(img_original: npt.NDArray[np.uint8], img_inverted: npt.NDArray[np.uint8], eps: float = 0.99) -> npt.NDArray[np.uint8]:
	img_original_float = img_original.astype(np.float32)
	img_inverted_float = img_inverted.astype(np.float32)
	img_original_normalized = img_original_float / float(255)
	img_inverted_normalized = img_inverted_float / float(255)
	img_inverted_clipped = np.clip(img_inverted_normalized, 0, eps)
	result = img_original_normalized / (1 - img_inverted_clipped)
	return np.clip(result * 255, 0, 255).astype(np.uint8)


def generate_sketch_from_array(img: npt.NDArray[np.uint8], blur_strength: float = 0.05) -> npt.NDArray[np.uint8]:
    inverted = invert_img(img)
    blurred = gaussian_blur(inverted, blur_strength)
    return color_dodge_blend(img, blurred)


def generate_sketch(img_path: str) -> npt.NDArray[np.uint8]:
	grayscale = convert_to_grayscale(img_path)
	sketch = generate_sketch_from_array(grayscale)
	return sketch


def main() -> None:
	path = input("Enter the path to the image: ")
	sketch = generate_sketch(path)
	while True:
		save = input("Save? [Y]es, [N]o: ").strip().lower()
		if save == "y":
			filepath = input("Enter output filepath: ")
			success = cv2.imwrite(filepath, sketch)
			if success:
				print(f"Image successfully saved as {filepath}")
			else:
				print("Error saving image")
			break
		elif save == "n":
			cv2.imshow("Sketch", sketch)
			cv2.waitKey(0)
			break


if __name__ == "__main__":
	main()
#	generate_sketch("test_img.jpeg")
