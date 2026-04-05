import torch
import torchvulkan as torchvk

if __name__ == "__main__":
    try:
        assert torchvk.is_available()
        print(f"torchvulkan is available! Found {torchvk.device_count()} device(s).")

    except Exception as e:
        print(f"An error occurred while checking torchvulkan availability: {e}")