import torch
import torch.nn as nn

class AttentionBoundingBoxLoss(nn.Module):
    def __init__(self):
        super(AttentionBoundingBoxLoss, self).__init__()

    def forward(self, attention_map, bounding_box):
        """
        Compute the loss based on the attention map and bounding box.

        Parameters:
        attention_map (torch.Tensor): The attention map of shape (H, W).
        bounding_box (tuple): The bounding box coordinates (x_min, y_min, x_max, y_max).

        Returns:
        torch.Tensor: The computed loss.
        """
        x_min, y_min, x_max, y_max = bounding_box

        # Create a mask for the bounding box area
        mask = torch.zeros_like(attention_map)
        mask[y_min:y_max, x_min:x_max] = 1

        # Compute the attention values inside and outside the bounding box
        attention_inside = attention_map * mask
        attention_outside = attention_map * (1 - mask)

        # Calculate the loss as the sum of attention values outside the bounding box
        loss = attention_outside.sum()

        return loss

class AttentionBoundingBoxLossScaledDistance(nn.Module):
    def __init__(self):
        super(AttentionBoundingBoxLossScaledDistance, self).__init__()

    def forward(self, attention_map, bounding_box):
        """
        Compute the loss based on the attention map and bounding box, scaled by the distance from the bounding box.

        Parameters:
        attention_map (torch.Tensor): The attention map of shape (H, W).
        bounding_box (tuple): The bounding box coordinates (x_min, y_min, x_max, y_max).

        Returns:
        torch.Tensor: The computed loss.
        """
        x_min, y_min, x_max, y_max = bounding_box
        H, W = attention_map.shape

        # Create a mask for the bounding box area
        mask = torch.zeros_like(attention_map)
        mask[y_min:y_max, x_min:x_max] = 1

        # Compute the attention values outside the bounding box
        attention_outside = attention_map * (1 - mask)

        # Calculate the distance of each pixel from the bounding box
        y_indices, x_indices = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        x_dist = torch.min(torch.abs(x_indices - x_min), torch.abs(x_indices - x_max))
        y_dist = torch.min(torch.abs(y_indices - y_min), torch.abs(y_indices - y_max))
        distance = torch.sqrt(x_dist**2 + y_dist**2)

        # Scale the attention values outside the bounding box by their distance
        scaled_attention_outside = attention_outside * distance

        # Calculate the loss as the sum of scaled attention values outside the bounding box
        loss = scaled_attention_outside.sum()

        return loss

# Example usage
attention_map = torch.rand(224, 224)  # Example attention map
bounding_box = (50, 50, 150, 150)  # Example bounding box coordinates

loss = AttentionBoundingBoxLoss()
loss_value = loss(attention_map, bounding_box)
print('Loss value:', loss_value)

loss_fn_dist = AttentionBoundingBoxLossScaledDistance()
loss_dist = loss_fn_dist(attention_map, bounding_box)
print('Loss value with scaled distance:', loss_dist)