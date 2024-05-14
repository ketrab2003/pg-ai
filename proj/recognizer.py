import torch
from main import Net, transform

# load the model
network = Net()
network.load_state_dict(torch.load('model.pth'))

# interactive mode
# let user provide an image and get the prediction

from PIL import Image

def recognize_digit_internal(image: Image):
    image = image.convert('L').resize((28, 28))
    image = transform(image)
    image = image.view(1, 1, 28, 28)
    output = network(image)
    pred = output.data.max(1, keepdim=True)[1]
    return pred.item()

def recognize_digit(image: Image):
    # repeat 10 times and take the most common result
    predictions = [recognize_digit_internal(image) for _ in range(10)]
    return max(set(predictions), key=predictions.count)

if __name__ == '__main__':
    digit = recognize_digit(Image.open('test_image.png'))
    print(f'The digit in the image is: {digit}')
