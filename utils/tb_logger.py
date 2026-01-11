from tensorboardX import SummaryWriter

class TensorBoardLogger:
    def __init__(self, log_dir):
        self.summary_writer = SummaryWriter(log_dir)

    def add_scalars(self, tag_step_value_dict):
        """
        :param parent_tag: str, e.g. "Training Loss"
        :param tag_step_value_dict: dict, e.g., {"key":(step, value), "q_grad":(10000, 1.11)}
        """
        for tag, (step, value) in tag_step_value_dict.items():
            self.summary_writer.add_scalar(tag, value, step)

    def add_images(self, tag_image_dict):
        """
        :param tag_image_dict: dict, where the key is the tag (str) and the value is a tuple of (step, image_tensor)
        The image_tensor should be a tensor representing an image or a batch of images in the appropriate format
        (e.g., torch.Tensor with shape [batch_size, channels, height, width] for PyTorch).
        """
        for tag, (step, image_tensor) in tag_image_dict.items():
            self.summary_writer.add_image(tag, image_tensor, step)
