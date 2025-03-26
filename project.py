import cv2
import numpy as np

# Helper functions for binary conversion
def to_binary(data):
    if isinstance(data, str):
        return ''.join(format(ord(c), '08b') for c in data)
    elif isinstance(data, bytes):
        return ''.join(format(byte, '08b') for byte in data)
    elif isinstance(data, np.ndarray):
        return ''.join(format(byte, '08b') for byte in data.flatten())
    elif isinstance(data, int):
        return format(data, '08b')
    else:
        raise TypeError("Unsupported data type for conversion to binary.")

def from_binary(binary_data):
    chars = [chr(int(binary_data[i:i+8], 2)) for i in range(0, len(binary_data), 8)]
    return ''.join(chars)

def encode_text_in_image(image_path, secret_text, output_image_path):
    image = cv2.imread(image_path)
    binary_secret = to_binary(secret_text) + '1111111111111110'  # End delimiter
    secret_len = len(binary_secret)

    i = 0
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            for channel in range(image.shape[2]):
                if i < secret_len:
                    image[row, col, channel] = (image[row, col, channel] & 0xFE) | int(binary_secret[i])
                    i += 1

    cv2.imwrite(output_image_path, image)
    print(f"Text hidden in image saved to {output_image_path}")

def decode_text_from_image(stego_image_path):
    stego_image = cv2.imread(stego_image_path)
    binary_data = ""

    for row in range(stego_image.shape[0]):
        for col in range(stego_image.shape[1]):
            for channel in range(stego_image.shape[2]):
                binary_data += str(stego_image[row, col, channel] & 1)

    delimiter_idx = binary_data.find('1111111111111110')
    secret_text = from_binary(binary_data[:delimiter_idx])
    print(f"Decoded text: {secret_text}")

def hide_text_in_video_hs(video_path, secret_text, output_video_path):
    video = cv2.VideoCapture(video_path)
    binary_secret = to_binary(secret_text) + '1111111111111110'  # Delimiter
    secret_len = len(binary_secret)

    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frame_size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    i = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        for row in range(frame.shape[0]):
            for col in range(frame.shape[1]):
                for channel in range(frame.shape[2]):
                    if i < secret_len:
                        pixel_value = frame[row, col, channel]
                        if pixel_value % 2 == 0:
                            frame[row, col, channel] += 1 if binary_secret[i] == '1' else 0
                        else:
                            frame[row, col, channel] -= 1 if binary_secret[i] == '0' else 0
                        i += 1

        out.write(frame)

    video.release()
    out.release()
    print(f"Text hidden in video saved to {output_video_path}")

def decode_text_from_video_hs(stego_video_path):
    video = cv2.VideoCapture(stego_video_path)
    binary_data = ""

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        for row in range(frame.shape[0]):
            for col in range(frame.shape[1]):
                for channel in range(frame.shape[2]):
                    binary_data += str(frame[row, col, channel] % 2)

    video.release()
    delimiter_idx = binary_data.find('1111111111111110')
    if delimiter_idx == -1:
        print("No hidden text found.")
        return

    secret_text = from_binary(binary_data[:delimiter_idx])
    print(f"Decoded text: {secret_text}")

def hide_image_in_image(cover_image_path, secret_image_path, output_image_path):
    cover_image = cv2.imread(cover_image_path)
    secret_image = cv2.imread(secret_image_path)
    

    secret_image = cv2.resize(secret_image, (cover_image.shape[1], cover_image.shape[0]))
    
    if cover_image.shape[2] != secret_image.shape[2]:
        raise ValueError("The cover image and the secret image must have the same number of channels.")
    

    cover_image = np.array(cover_image, dtype=np.uint8)
    secret_image = np.array(secret_image, dtype=np.uint8)


    for i in range(cover_image.shape[0]):
        for j in range(cover_image.shape[1]):
            for c in range(cover_image.shape[2]):
            
                cover_image[i, j, c] = (cover_image[i, j, c] & 0xFE) | (secret_image[i, j, c] >> 7)
    
    # Save the encoded image
    cv2.imwrite(output_image_path, cover_image)
    print(f"Encoded image saved to: {output_image_path}")

def decode_image_from_image(stego_image_path, output_image_path):
    # Load the stego image
    stego_image = cv2.imread(stego_image_path)
    
    # Extract the secret image from the stego image
    secret_image = np.zeros_like(stego_image)
    
    for i in range(stego_image.shape[0]):
        for j in range(stego_image.shape[1]):
            for c in range(stego_image.shape[2]):
                
                secret_image[i, j, c] = (stego_image[i, j, c] & 0x01) << 7
    
    # Save the extracted secret image
    cv2.imwrite(output_image_path, secret_image)
    print(f"Decoded image saved to: {output_image_path}")


def hide_image_in_image_dct(cover_image_path, secret_image_path, output_image_path):
    cover_image = cv2.imread(cover_image_path, cv2.IMREAD_COLOR)
    secret_image = cv2.imread(secret_image_path, cv2.IMREAD_COLOR)

    # Resize secret image to fit the cover image
    secret_image = cv2.resize(secret_image, (cover_image.shape[1], cover_image.shape[0]))

    # Convert images to float32 for DCT
    cover_image_float = np.float32(cover_image) / 255.0
    secret_image_float = np.float32(secret_image) / 255.0

    # Perform 2D DCT on each channel of the cover image
    cover_dct = [cv2.dct(cover_image_float[:, :, c]) for c in range(3)]

    # Embed secret image in the LL part of the DCT
    for c in range(3):
        cover_dct[c][:, :] += 0.01 * secret_image_float[:, :, c]

    # Perform inverse DCT
    stego_image = np.zeros_like(cover_image_float)
    for c in range(3):
        stego_image[:, :, c] = cv2.idct(cover_dct[c])

    # Convert back to uint8 and save
    stego_image = np.uint8(np.clip(stego_image * 255.0, 0, 255))
    cv2.imwrite(output_image_path, stego_image)
    print(f"Image hidden using DCT. Saved to {output_image_path}")

def decode_image_from_image_dct(stego_image_path, output_image_path):
    stego_image = cv2.imread(stego_image_path, cv2.IMREAD_COLOR)
    stego_image_float = np.float32(stego_image) / 255.0

    # Perform 2D DCT on each channel
    stego_dct = [cv2.dct(stego_image_float[:, :, c]) for c in range(3)]

    # Extract secret image from LL part
    secret_image = np.zeros_like(stego_image_float)
    for c in range(3):
        secret_image[:, :, c] = stego_dct[c][:, :] / 0.01

    # Convert back to uint8 and save
    secret_image = np.uint8(np.clip(secret_image * 255.0, 0, 255))
    cv2.imwrite(output_image_path, secret_image)
    print(f"Decoded secret image saved to {output_image_path}")


    
# Hiding and decoding an image in a video
def hide_image_in_video(video_path, secret_image_path, output_video_path):
    video = cv2.VideoCapture(video_path)
    secret_image = cv2.imread(secret_image_path)

    secret_shape_binary = to_binary(secret_image.shape[0]) + to_binary(secret_image.shape[1]) + to_binary(secret_image.shape[2])

    secret_image_resized = cv2.resize(secret_image, (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    secret_binary = ''.join(format(pixel, '08b') for pixel in secret_image_resized.flatten())
    combined_binary = secret_shape_binary + secret_binary + '1111111111111110'
    secret_len = len(combined_binary)
    i = 0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frame_size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        for row in range(frame.shape[0]):
            for col in range(frame.shape[1]):
                for channel in range(frame.shape[2]):
                    if i < secret_len:
                        frame[row, col, channel] = (frame[row, col, channel] & 0xFE) | int(combined_binary[i])
                        i += 1

        out.write(frame)

    video.release()
    out.release()
    print(f"Image hidden in video saved to {output_video_path}")

def decode_image_from_video(stego_video_path, output_image_path):
    video = cv2.VideoCapture(stego_video_path)
    binary_data = ""

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        for row in range(frame.shape[0]):
            for col in range(frame.shape[1]):
                for channel in range(frame.shape[2]):
                    binary_data += str(frame[row, col, channel] & 1)

    video.release()

    delimiter_idx = binary_data.find('1111111111111110')
    secret_binary = binary_data[:delimiter_idx]

    secret_shape_binary = secret_binary[:24]
    height = int(secret_shape_binary[:8], 2)
    width = int(secret_shape_binary[8:16], 2)
    channels = int(secret_shape_binary[16:24], 2)

    secret_data = [int(secret_binary[i:i+8], 2) for i in range(24, len(secret_binary), 8)]
    secret_image = np.array(secret_data, dtype=np.uint8).reshape((height, width, channels))

    cv2.imwrite(output_image_path, secret_image)
    print(f"Decoded secret image saved to {output_image_path}")

# Main Menu
def main():
    while True:
        print("\nChoose an option:")
        print("1. Hide text in image")
        print("2. Decode text from image")
        print("3. Hide text in video")
        print("4. Decode text from video")
        print("5. Hide image in image")
        print("6. Decode image from image")
        print("7. Hide image in video")
        print("8. Decode image from video")
        print("9. Hide image in image (DCT)[It is irreversible]")
        print("10. Decode image from image (DCT)")
        print("11. Exit")

        option = input("Enter option: ")

        if option == "1":
            image_path = input("Enter the image path: ")
            secret_text = input("Enter the secret text: ")
            output_image_path = input("Enter the output image path: ")
            encode_text_in_image(image_path, secret_text, output_image_path)

        elif option == "2":
            stego_image_path = input("Enter the steganographed image path: ")
            decode_text_from_image(stego_image_path)

        elif option == "3":
            video_path = input("Enter the video path: ")
            secret_text = input("Enter the secret text: ")
            output_video_path = input("Enter the output video path: ")
            hide_text_in_video_hs(video_path, secret_text, output_video_path)

        elif option == "4":
            stego_video_path = input("Enter the steganographed video path: ")
            decode_text_from_video_hs(stego_video_path)

        elif option == "5":
            cover_image_path = input("Enter the cover image path: ")
            secret_image_path = input("Enter the secret image path: ")
            output_image_path = input("Enter the output image path: ")
            hide_image_in_image(cover_image_path, secret_image_path, output_image_path)

        elif option == "6":
            stego_image_path = input("Enter the steganographed image path: ")
            output_image_path = input("Enter the output image path: ")
            decode_image_from_image(stego_image_path, output_image_path)

        elif option == "7":
            video_path = input("Enter the video path: ")
            secret_image_path = input("Enter the secret image path: ")
            output_video_path = input("Enter the output video path: ")
            hide_image_in_video(video_path, secret_image_path, output_video_path)

        elif option == "8":
            stego_video_path = input("Enter the steganographed video path: ")
            output_image_path = input("Enter the output image path: ")
            secret_shape = tuple(map(int, input("Enter secret image shape as 'height,width,channels': ").split(',')))
            decode_image_from_video(stego_video_path, output_image_path, secret_shape)

        elif option == "9":
            cover_image_path = input("Enter the cover image path: ")
            secret_image_path = input("Enter the secret image path: ")
            output_image_path = input("Enter the output image path: ")
            hide_image_in_image_dct(cover_image_path, secret_image_path, output_image_path)

        elif option == "10":
            stego_image_path = input("Enter the steganographed image path: ")
            output_image_path = input("Enter the output image path: ")
            decode_image_from_image_dct(stego_image_path, output_image_path)    

        elif option == "11":
            break

        else:
            print("Invalid option. Try again.")

if __name__ == "__main__":
    main()
