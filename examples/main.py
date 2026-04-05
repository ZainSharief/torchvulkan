import torch
import torchvulkan as torchvk

if __name__ == "__main__":
    try:
        assert torchvk.is_available()
        print("torchvulkan is available!")

    except Exception as e:
        print(f"An error occurred while checking torchvulkan availability: {e}")